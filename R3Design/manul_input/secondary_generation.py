import subprocess
import sys
import os
import os.path as osp
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)

from Bio.PDB import PDBParser
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
root_dir = os.path.dirname(parent_dir)
def call_python2_script(pdb_path, chain_name):
    """
    Call a Python 2 script using Moderna to get the secondary structure from a different Conda environment.
    """
    script_path = osp.join(osp.dirname(osp.dirname(__file__)), 'manul_input', 'get_secondary_structure.py')
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

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py <pdb_path> <chain_name>, using default setting")
        pdb_path = osp.join(root_dir, 'RDesign/example/3j16_L_1_72/native.pdb')
        chain_name = 'A'
    else:
        pdb_path = sys.argv[1]
        chain_name = sys.argv[2]

    # Get secondary structure from Moderna
    moderna_structure = call_python2_script(pdb_path, chain_name)
    if moderna_structure is None:
        print("Failed to get secondary structure from Moderna.")
        sys.exit()


    parser = PDBParser()
    structure = parser.get_structure('RNA', pdb_path)
    model = structure[0]  # Get the first model
    sequence = ''
    for chain in model:
        if chain.id == chain_name:
            for residue in chain:
                if residue.get_resname() in ['A', 'C', 'G', 'U']:
                    sequence += {'A':'A', 'C':'C', 'G':'G', 'U':'U'}[residue.get_resname()]
            break
    
    if not sequence:
        print(f"No sequence found for chain {chain_name} in {pdb_path}")
        sys.exit()

    # Predict structure using RNAfold
    rnafold_structure = predict_rnafold(sequence)

    # Calculate accuracy
    if len(moderna_structure) != len(rnafold_structure):
        print("Error: Length mismatch between Moderna and RNAfold predictions.")
        sys.exit()
    
    true_list = list(moderna_structure)  # Moderna's output as the 'true' structure
    pred_list = list(rnafold_structure)  # RNAfold's output as the 'predicted' structure
    accuracy = accuracy_score(true_list, pred_list)
    
    print(f"Sequence: {sequence}")
    print(f"Moderna Structure: {moderna_structure}")
    print(f"RNAfold Structure: {rnafold_structure}")
    print(f"Accuracy: {accuracy:.4f}")

# import subprocess
# import sys
# import os
# import os.path as osp
# current_dir = os.path.dirname(__file__)
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
# root_dir = os.path.dirname(parent_dir)

# def call_python2_script(pdb_path,chain_name):
#     """
#     Call a Python 2 script from a different Conda environment.
#     """
#     script_path = os.path.join(parent_dir, 'manul_input', 'get_secondary_structure.py')
#     command = ['conda', 'run', '-n', 'moderna', 'python', script_path, pdb_path]
#     try:
#         result = subprocess.check_output(command, universal_newlines=True)
#         return result.strip()
#     except subprocess.CalledProcessError as e:
#         print("Error calling Python 2 script:", e)
#         return None
    
# if __name__ == '__main__':
#     if len(sys.argv) < 3:
#         print("Usage: python script.py <pdb_path> <chain_name>, using default setting")
#         pdb_path = osp.join(root_dir, 'RDesign/example/3j16_L_1_72/refined_model.pdb')
#         chain_name = 'A'
#     else:
#         pdb_path = sys.argv[1]
#         chain_name = sys.argv[2]
#     # pdb_path = osp.join(root_dir, 'RDesign/example/3wbm_rna_X.pdb')
#     # chain_name = 'X'
#     result = call_python2_script(pdb_path, chain_name)
#     print(result)
