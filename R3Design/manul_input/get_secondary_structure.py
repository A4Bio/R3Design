# get_secondary_structure.py
import os
from moderna import *

def get_secondary_structure(pdb_path, chain):
    """
    Load a PDB file and return its secondary structure.
    """
    try:
        t = load_template(pdb_path,chain)
        clean_structure(t)
        get_secstruc(t)
        return get_secstruc(t)
    except Exception as e:
        print "Error processing {}: {}".format(pdb_path, e)
        return None

if __name__ == "__main__":
    import sys
    pdb_path = sys.argv[1]
    chain_name = sys.argv[2]
    # pdb_path = '/tancheng/zyj/RDesign+/RDesign/example/4wb2_rna_D.pdb'
    # chain_name = 'D'
    ss = get_secondary_structure(pdb_path,chain_name)
    print(ss)
