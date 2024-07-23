#!/bin/bash
set -e

source /root/anaconda3/etc/profile.d/conda.sh


conda activate RF2NA

echo "conda activate RF2NA"
# make the script stop when error (non-true exit code) occurs

# Check if the RNA .fa file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <rna-file.fa>"
    exit 1
fi

# RNA .fa file
RNA_FILE=$1

# Path to the run_rna.sh script
# RUN_RNA_SCRIPT="/tancheng/zyj/RoseTTAFold2NA/run_RF2NA.sh"
# RUN_RNA_SCRIPT="../../RoseTTAFold2NA/run_RF2NA.sh"
# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# 构建目标脚本的绝对路径
RUN_RNA_SCRIPT="$SCRIPT_DIR/../../RoseTTAFold2NA/run_RF2NA.sh"

# Check if run_rna.sh script exists
if [ ! -f "$RUN_RNA_SCRIPT" ]; then
    echo "run_rna.sh script not found in the current directory."
    exit 1
fi

# Check if the RNA file exists
if [ ! -f "$RNA_FILE" ]; then
    echo "RNA file $RNA_FILE does not exist."
    exit 1
fi

# Process the provided .fa file
orig_dir=$(dirname "$RNA_FILE")
base_name=$(basename "$RNA_FILE" .fa)
result_dir="${orig_dir}/result/${base_name}"

if [ ! -d "$result_dir" ]; then
    mkdir -p "$result_dir"
fi

output_dir="$result_dir"

echo "Processing $RNA_FILE with output directory $output_dir..."
bash "$RUN_RNA_SCRIPT" "$output_dir" "R:$RNA_FILE"

core_files=("$output_dir"/core*)
if [ -f "${core_files[0]}" ]; then
    echo "Removing core files in $output_dir..."
    rm "$output_dir"/core*
fi

echo "Processing complete."
