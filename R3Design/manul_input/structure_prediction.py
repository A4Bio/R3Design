import subprocess
import os.path as osp
import re
import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
root_dir = os.path.dirname(parent_dir)

# from whole_pipeline import structure_predict
def extract_pdb_name(pdb_path):
    pdb_file_name = osp.basename(pdb_path)  # Extract the file name
    pdb_name, _ = osp.splitext(pdb_file_name)  # Remove the file extension
    return pdb_name
    
def metric_evaluation(ori_pdb_path, target_pdb_path):
    # Extract PDB names from the provided paths
    ori_pdb_name = extract_pdb_name(ori_pdb_path)
    target_pdb_name = extract_pdb_name(target_pdb_path)
    # Construct the path for the native PDB file
    if osp.exists(ori_pdb_path) and osp.exists(target_pdb_path):
        try:
            # Running the subprocess and capturing its output
            us_path = osp.join(parent_dir,'utils/USalign')
            printed_log = subprocess.Popen([us_path, target_pdb_path, ori_pdb_path], 
                                           stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
            # Extract scores from the output
            scores = re.findall('TM-score= [0-9]*\.?[0-9]*', str(printed_log))
            score1, score2 = [float(score.split('TM-score=')[1]) for score in scores]
            avg_score = 0.5*(score1 + score2)

            # Extract RMSD from the output
            rmsd_match = re.search('RMSD= +([0-9]*\.?[0-9]*)', str(printed_log))
            rmsd = float(rmsd_match.group(1)) if rmsd_match else 'N/A'

            return [ori_pdb_name, target_pdb_name, avg_score, rmsd]
        except Exception as e:
            return [ori_pdb_name, target_pdb_name, 'Error', 'Error']
    else:
        return [ori_pdb_name, target_pdb_name,  'PDB not found', 'N/A']
    
    
    
def structure_predict(rna_file_path):
    # Ensure the RNA file path is a valid string
    if not isinstance(rna_file_path, str):
        print("Invalid RNA file path.")
        return

    # Specify the path to your shell script
    script_path = os.path.join(parent_dir,'manul_input/run_single.sh')

    try:
        subprocess.run(["bash", script_path, rna_file_path], check=True)
        print("Shell script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def main(rna_file_path=osp.join(root_dir,'RoseTTAFold2NA/example/RNA.fa'), ori_pdb_path=osp.join(root_dir,'RDesign/example/native.pdb')):
    structure_predict(rna_file_path)
    rna_file_directory = osp.dirname(rna_file_path)
    rna_name = extract_pdb_name(rna_file_path)
    target_pdb_path = osp.join(rna_file_directory,'result',rna_name,"models/model_00.pdb")
    metric_got = metric_evaluation(ori_pdb_path, target_pdb_path)
    print(f"Original PDB: {metric_got[0]}, Target PDB: {metric_got[1]}, \
                TM-Score: {metric_got[2]}, RMSD: {metric_got[3]}\n")
    

if __name__ == '__main__':
    # Example usage: python script.py /path/to/rna/file
    if len(sys.argv) < 3:
        print("Usage: <rna_file_path>, <ori_pdb_path>, using default setting")
        print("Running example sequence")
        # pdb_file_path = osp.join(parent_dir,'example/4tux_rna_D.pdb') 
        # target_pdb_path = osp.join(parent_dir,'example/4tux_rna_D.pdb')
        main()
    else:
        main(sys.argv[1], sys.argv[2])
    
