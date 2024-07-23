import sys 
import os
import os.path as osp
from datetime import datetime
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
root_dir = os.path.dirname(parent_dir)
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from structure_prediction import structure_predict,metric_evaluation
from sequence_design import eval_sequence, process_single_pdb

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdb"):
            pdb_file_path = osp.join(directory_path, filename)
            target_pdb_path = pdb_file_path  # In this script, target is the same as input
            process_pdb_file(pdb_file_path, target_pdb_path)

def process_pdb_file(pdb_file_path, target_pdb_path):
    parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    res_dir = osp.join(parent_dir, 'manul_input/full_result')
    script_path = osp.join(parent_dir, 'manul_input/run_single.sh')

    processed_data = process_single_pdb(pdb_file_path)
    name_lst, f1s, rec_lst, S_preds_lst, S_trues_lst = eval_sequence(processed_data)
    
    for name, f1, rec, s_pred, s_true in zip(name_lst, f1s, rec_lst, S_preds_lst, S_trues_lst):
        folder_path = osp.join(res_dir, name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pred_file_path = osp.join(folder_path, name + '_pred.fa')
        pred_record = SeqRecord(Seq(s_pred), id=name)
        with open(pred_file_path, "w") as fasta_file:
            SeqIO.write(pred_record, fasta_file, "fasta")

        true_file_path = osp.join(folder_path, name + '_true.fa')
        true_record = SeqRecord(Seq(s_true), id=name)
        with open(true_file_path, "w") as fasta_file:
            SeqIO.write(true_record, fasta_file, "fasta")

        log_file_path = osp.join(folder_path, 'log.txt')
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{current_time} - Name: {name}, F1: {f1}, Rec: {rec}, Predicted: {s_pred}, True: {s_true}\n")

        structure_predict(pred_file_path)
        pred_file_directory = osp.dirname(pred_file_path)
        pred_pdb_path = osp.join(pred_file_directory, 'result', name + '_pred', "models/model_00.pdb")
        metric_got = metric_evaluation(pred_pdb_path, target_pdb_path)
        
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Original PDB: {metric_got[0]}, Target PDB: {metric_got[1]}, TM-Score: {metric_got[2]}, RMSD: {metric_got[3]}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory_of_PDB_files>")
        directory_path = 'aptamer_exp/RNA_only/extracted_structures'
    else:
        directory_path = sys.argv[1]
    process_directory(directory_path)
