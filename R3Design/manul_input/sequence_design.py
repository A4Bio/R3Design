from Bio.PDB import PDBParser
import numpy as np
import os
import os.path as osp
import subprocess
import torch
import numpy as np
import sys 
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
root_dir = os.path.dirname(parent_dir)
import json
import argparse
from exp import Exp
from methods.utils import cuda
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F

def call_python2_script(pdb_path,chain_name):
    """
    Call a Python 2 script from a different Conda environment.
    """
    script_path = os.path.join(parent_dir, 'manul_input', 'get_secondary_structure.py')
    command = ['conda', 'run', '-n', 'moderna', 'python', script_path, pdb_path, chain_name]
    try:
        result = subprocess.check_output(command, universal_newlines=True)
        return result.strip()
    except subprocess.CalledProcessError as e:
        print("Error calling Python 2 script:", e)
        return None


def predict_rnafold(sequence):
    """
    Predicts RNA secondary structure using RNAfold.
    """
    result = subprocess.run(['RNAfold'], input=sequence, text=True, capture_output=True)
    output_lines = result.stdout.strip().split('\n')
    if len(output_lines) > 1:
        return output_lines[1].split(' ')[0]  # Extract the secondary structure, ignoring the free energy
    return ""

def process_single_pdb(pdb_file):
    backbone_atoms = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]
    alphabet_set = 'AUCG'

    pdb_name = osp.basename(pdb_file).split('.')[0]
    parser = PDBParser()
    structure = parser.get_structure('', pdb_file)
    coords = {
        'P': [], "O5'": [], "C5'": [], "C4'": [], "C3'": [], "O3'": []
    }


    for model in structure:
        chain = list(model.get_chains())[0]
        chain_name = chain.id

        seq = ''  
        coords_dict = {atom_name: [np.nan, np.nan, np.nan] for atom_name in backbone_atoms}

        for residue in chain:
            seq += residue.get_resname()

            for atom in residue:
                if atom.name in backbone_atoms:
                    coords_dict[atom.name] = atom.get_coord()

            list(map(lambda atom_name: coords[atom_name].append(list(coords_dict[atom_name])), backbone_atoms))


        for atom_name in backbone_atoms:
            assert len(seq) == len(coords[atom_name]), f"Length of sequence {len(seq)} and coordinates {len(coords[atom_name])} for {atom_name} do not match."


        bad_chars = set(seq).difference(alphabet_set)
        if len(bad_chars) != 0:
            print('Found bad characters in sequence:', bad_chars)

        break 
    ss = call_python2_script(pdb_file,chain_name)
    rnafold_structure = predict_rnafold(seq)
    data = {
        'seq': seq,
        'coords': coords,
        'chain_name': chain_name,
        'name': pdb_name,
        'ss': ss,
        'pred_ss': rnafold_structure
    }

    return data

def find_bracket_pairs(ss, seq):
    pairs = []
    stack = []
    for i, c in enumerate(ss):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                pairs.append((stack.pop(), i))
            else:
                pairs.append((None, i)) 
    if stack:
        pairs.extend(zip(stack[::-1], range(i, i - len(stack), -1)))
        
    npairs = []
    for pair in pairs:
        if None in pair:
            continue
        p_a, p_b = pair
        if (seq[p_a], seq[p_b]) in (('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C')):
            npairs.append(pair)
        # else:
        #     print('error')
    return npairs

def featurize_HC(batch):
    """ Pack and pad batch into torch tensors """
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    # print(L_max)
    # L_max = 2000
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    clus = np.zeros([B], dtype=np.int32)
    ss_pos = np.zeros([B, L_max], dtype=np.int32)
    
    ss_pair = []
    names = []

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']], 1)
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        ss_pos[i, :l] = np.asarray([1 if ss_val!='.' else 0 for ss_val in b['ss']], dtype=np.int32)
        ss_pair.append(find_bracket_pairs(b['ss'], b['seq']))
        names.append(b['name'])
        
        clus[i] = i

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    clus = torch.from_numpy(clus).to(dtype=torch.long)
    return X, S, mask, lengths, clus, ss_pos, ss_pair, names

def eval_sequence(processed_data):
    
    pre_base_pairs = {0: 1, 1: 0, 2: 3, 3: 2}
    pre_great_pairs = ((0, 1), (1, 0), (2, 3), (3, 2))

    svpath = osp.join(root_dir,'ckpt_path/')
    config = json.load(open(svpath+'model_param.json','r'))
    args = argparse.Namespace(**config)
    print(args)
    exp = Exp(args)
    exp.method.model.load_state_dict(torch.load(svpath+'checkpoint.pth'))
    exp.method.model.eval()


    alphabet = 'AUCG'
    S_preds, S_trues, name_lst, rec_lst = [], [], [], []
    S_preds_lst, S_trues_lst = [], []
    from API.single_file import Single_input

    single_data = Single_input(processed_data)

    f1s = []
    for idx, sample in enumerate(single_data):
        sample = featurize_HC([sample])
        X, S, mask, lengths, clus, ss_pos, ss_pair, names = sample
        X, S, mask, ss_pos = cuda((X, S, mask, ss_pos), device=exp.device)
        logits, gt_S = exp.method.model.sample(X=X, S=S, mask=mask)
        log_probs = F.log_softmax(logits, dim=-1)
        # secondary sharpen
        ss_pos = ss_pos[mask == 1].long()
        log_probs = log_probs.clone()
        log_probs[ss_pos] = log_probs[ss_pos] / exp.args.ss_temp
        S_pred = torch.argmax(log_probs, dim=1)
        
        pos_log_probs = log_probs.softmax(-1)
        for pair in ss_pair[0]:
            s_pos_a, s_pos_b = pair
            if s_pos_a == None or s_pos_b == None or s_pos_b >= S_pred.shape[0]:
                continue
            
            if (S_pred[s_pos_a].item(), S_pred[s_pos_b].item()) in pre_great_pairs:
                continue
            
            if pos_log_probs[s_pos_a][S_pred[s_pos_a]] > pos_log_probs[s_pos_b][S_pred[s_pos_b]]:
                S_pred[s_pos_b] = pre_base_pairs[S_pred[s_pos_a].item()]
            elif pos_log_probs[s_pos_a][S_pred[s_pos_a]] < pos_log_probs[s_pos_b][S_pred[s_pos_b]]:
                S_pred[s_pos_a] = pre_base_pairs[S_pred[s_pos_b].item()]

        _, _, f1, _ = precision_recall_fscore_support(S_pred.cpu().numpy().tolist(), gt_S.cpu().numpy().tolist(), average=None)
        f1s.append(f1.mean())

        S_preds += S_pred.cpu().numpy().tolist()
        S_trues += gt_S.cpu().numpy().tolist()

        S_preds_lst.append(''.join([alphabet[a_i] for a_i in S_pred.cpu().numpy().tolist()]))
        S_trues_lst.append(''.join([alphabet[a_i] for a_i in gt_S.cpu().numpy().tolist()]))
        name_lst.extend(names)

        cmp = S_pred.eq(gt_S)
        recovery_ = cmp.float().mean().cpu().numpy()
        rec_lst.append(recovery_)

    _, _, f1, _ = precision_recall_fscore_support(S_trues, S_preds, average=None)

    return name_lst, f1s, rec_lst, S_preds_lst, S_trues_lst

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <pdb_file_path>, using default setting")
        pdb_file_path = osp.join(root_dir, 'R3Design/example/4tux_rna_C.pdb')
    else:
        pdb_file_path = sys.argv[1]   
    processed_data = process_single_pdb(pdb_file_path)
    name_lst, f1s, rec_lst, S_preds_lst, S_trues_lst = eval_sequence(processed_data)
    for name, f1, rec, s_pred, s_true in zip(name_lst, f1s,rec_lst, S_preds_lst, S_trues_lst):
        if 1:
            print("----------Result------------")
            print("PDB_ID:", name)
            print("F1 Score:", f1,"Recovery Score:",rec)
            print("Predicted Sequence:",s_pred)
            print("True Sequence:",s_true)
    
