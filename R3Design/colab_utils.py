import sys
sys.path.append('/content/R3Design/R3Design/')
import torch
import json
import argparse
from exp import Exp
from tqdm import tqdm
from methods.utils import cuda
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os.path as osp
import torch.nn.functional as F
from Bio.PDB import PDBParser, MMCIFParser

import py3Dmol
import os
import requests


def process_single_pdb(file_path, chain_name=None):
    backbone_atoms = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]
    alphabet_set = 'AUCG'

    file_name = osp.basename(file_path)
    file_extension = file_name.split('.')[-1].lower()
    structure_name = file_name.split('.')[0]

    if file_extension == 'pdb':
        parser = PDBParser()
    elif file_extension == 'cif':
        parser = MMCIFParser()
    else:
        raise ValueError("Unsupported file format. Please provide a PDB or CIF file.")

    structure = parser.get_structure('', file_path)
    coords = {
        'P': [], "O5'": [], "C5'": [], "C4'": [], "C3'": [], "O3'": []
    }

    for model in structure:
        if chain_name is None:
            chain = list(model.get_chains())[0]
        else:
            chain = model[chain_name]

        seq = ''
        coords_dict = {atom_name: [np.nan, np.nan, np.nan] for atom_name in backbone_atoms}

        for residue in chain:
            if residue.id[0] == " ":
                seq += residue.get_resname()

            for atom in residue:
                if atom.name in backbone_atoms:
                    coords_dict[atom.name] = atom.get_coord()

            list(map(lambda atom_name: coords[atom_name].append(list(coords_dict[atom_name])), backbone_atoms))

        for atom_name in backbone_atoms:
            assert len(seq) == len(coords[atom_name]), f"Length of sequence and coordinates for {atom_name} do not match."

        bad_chars = set(seq).difference(alphabet_set)
        if len(bad_chars) != 0:
            print('Found bad characters in sequence:', bad_chars)

        break

    data = {
        'seq': seq,
        'coords': coords,
        'chain_name': chain.id,
        'name': structure_name
    }

    return data


def featurize_HC(batch):
    """ Pack and pad batch into torch tensors """
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    clus = np.zeros([B], dtype=np.int32)
    ss_pos = np.zeros([B, L_max], dtype=np.int32)

    ss_pair = []
    names = []

    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']], 1)

        l = len(b['seq'])
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        names.append(b['name'])

        clus[i] = i

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    numbers = np.sum(mask, axis=1).astype(int)
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
    return X, S, mask, lengths, clus, names

def eval_sequence(exp,data):
  alphabet = 'AUCG'
  S_preds, S_trues, name_lst, rec_lst = [], [], [], []
  S_preds_lst, S_trues_lst = [], []
  for idx, sample in enumerate(data):
      sample = featurize_HC([sample])
      X, S, mask, lengths, clus, names = sample
      X, S, mask = cuda((X, S, mask), device=exp.device)
      logits, gt_S = exp.method.model.sample(X=X, S=S, mask=mask)
      log_probs = F.log_softmax(logits, dim=-1)
      S_pred = torch.argmax(log_probs, dim=1)

      S_preds += S_pred.cpu().numpy().tolist()
      S_trues += gt_S.cpu().numpy().tolist()

      S_preds_lst.append(''.join([alphabet[a_i] for a_i in S_pred.cpu().numpy().tolist()]))
      S_trues_lst.append(''.join([alphabet[a_i] for a_i in gt_S.cpu().numpy().tolist()]))
      name_lst.extend(names)

      cmp = S_pred.eq(gt_S)
      recovery_ = cmp.float().mean().cpu().numpy()
      rec_lst.append(recovery_)

  _, _, f1, _ = precision_recall_fscore_support(S_trues, S_preds, average=None)

  return name_lst, f1, rec_lst, S_preds_lst, S_trues_lst

def highlight_differences(pred_seq, true_seq):
    highlighted_pred = []
    highlighted_true = []

    for pred_char, true_char in zip(pred_seq, true_seq):
        if pred_char == true_char:
            highlighted_pred.append(pred_char)
            highlighted_true.append(true_char)
        else:
            # ANSI escape sequences for red text
            highlighted_pred.append(f'\033[91m{pred_char}\033[0m')
            highlighted_true.append(f'\033[91m{true_char}\033[0m')

    return ''.join(highlighted_pred), ''.join(highlighted_true)


def load_processed_data(single_data, pdb_file, dataset, chain_name=None):
    if single_data == "True":
        if chain_name == "":
            chain_name = None
        processed_data = [process_single_pdb(pdb_file, chain_name)]
    else:
        raise ValueError("Invalid input: Please provide a valid option for 'single_data' or 'dataset'.")
    return processed_data

def visualize_pdb(pdb_file):
    with open(pdb_file, 'r') as f:
        true_pdb = f.read()
    view = py3Dmol.view(width=400, height=300)
    view.addModel(true_pdb, 'pdb')
    view.setStyle({'model': 0}, {"cartoon": {"color": 'rgba(149,149,149,20)'}})
    view.zoomTo()
    view.show()

def print_results(single_data, name_lst, f1, rec_lst, S_preds_lst, S_trues_lst):
    if single_data == "True":
        print('Name:', name_lst[0])
        print('F1_Score:', np.mean(f1), 'Recovery:', np.mean(rec_lst))

        highlighted_pred_seq, highlighted_true_seq = highlight_differences(S_preds_lst[0], S_trues_lst[0])
        print('Predicted Sequence:\n' + highlighted_pred_seq)
        print('True Sequence:\n' + highlighted_true_seq)
    else:
        print('F1_Score:', np.mean(f1), 'Recovery:', np.mean(rec_lst))

def inference(exp, single_data='True', pdb_file=None, dataset='test', chain_name=None):
    processed_data = load_processed_data(single_data, pdb_file, dataset, chain_name)

    if single_data == "True" and pdb_file:
        visualize_pdb(pdb_file)

    name_lst, f1, rec_lst, S_preds_lst, S_trues_lst = eval_sequence(exp, processed_data)
    print_results(single_data, name_lst, f1, rec_lst, S_preds_lst, S_trues_lst)


def download_structure(pdb_id, save_dir='./'):
    """Download PDB or CIF file, trying PDB first and falling back to CIF."""
    pdb_url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    cif_url = f'https://files.rcsb.org/download/{pdb_id}.cif'

    pdb_file = os.path.join(save_dir, f'{pdb_id}.pdb')
    cif_file = os.path.join(save_dir, f'{pdb_id}.cif')

    # Try downloading PDB file first
    response = requests.get(pdb_url)
    if response.status_code == 200:
        with open(pdb_file, 'wb') as f:
            f.write(response.content)
        print(f'{pdb_id}.pdb downloaded successfully.')
        return pdb_file, f'{pdb_id}.pdb'
    else:
        print(f'PDB file for {pdb_id} not found, trying CIF.')

    # If PDB file not found, try downloading CIF file
    response = requests.get(cif_url)
    if response.status_code == 200:
        with open(cif_file, 'wb') as f:
            f.write(response.content)
        print(f'{pdb_id}.cif downloaded successfully.')
        return cif_file, f'{pdb_id}.cif'
    else:
        raise Exception(f'Failed to download PDB or CIF file for: {pdb_id}')



