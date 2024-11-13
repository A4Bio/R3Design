# R3Design
<a href="https://colab.research.google.com/drive/1tAoUHY6w8WeweByY7TFwIyXPyGXA4lMW#scrollTo=gzAKWozrYdag" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Introduction

**R3Design** is a comprehensive evaluation system for RNA sequence design and prediction. The core part of R3Design is the sequence design model, a tertiary structure-based RNA sequence design model. In this model, we used Moderna to generate the secondary structure from RNA molecules. After generating the predicted sequence, we used RosettaFold to predict the RNA tertiary structure, which is to test the capability of our predicted sequence folding into the desired sequence. We also provided several APIs for each component of the function.


## Code Structure
```
R3Design
├── moderna
├── R3Design
├── RoseTTAFold2NA
|-- RNAFold
├── environment files
```
For R3Design Model, we mainly use `R3Design/manul_input` folder for producing results.
## Environment Setup 

Our system is based on three independent conda environments. The main environment is in the `environment.yml` file, the moderna is in `moderna.yml` file, and manual installation of [Moderna](https://genesilico.pl/moderna/) is needed. Besides, the conda environment for RosettaFold2 is in `RF2na-linux.yml`file. Also, `RNAFold` is required to evaluate the secondary structure.

### RNAFold Installation 

To install RNAFold, please refer to the [official website](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html)

After installing the package, then execute the following commands:
```shell
tar -zxvf ./compress_files/ViennaRNA-2.6.4.tar.gz
cd ViennaRNA-2.6.4
./configure
make
sudo make install
```

If there is issue with `xlocale.h`. Please try the following commands:
```shell
make clean
sudo apt-get update
sudo apt-get upgrade
apt install build-essential
apt install libgsl-dev
./configure
make
sudo make install
```

### Main Environment

```shell
conda env create -f environment.yml
```
### Moderna Environment 
```shell
conda env create -f moderna_env.yml
conda activate moderna
cd moderna/moderna_source_1.7.1
python setup.py install
```
### RosettaFold Environment 
The original installation step could be found [here](https://github.com/uw-ipd/RoseTTAFold2NA)

To fully install RosettaFold2NA, you should also download the required RNA datasets as the installation process provided in the original readme file.

```shell
conda env create -f RF2na-linux.yml
conda activate RF2NA
cd RoseTTAFold2NA
## Installing DGL according system's cuda version
# https://www.dgl.ai/pages/start.html
conda install -c dglteam dgl

#Install ZSTD
conda install ZSTD
cd SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
```

##  API
Our system could be split into four components: 
  * Secondary structure generation
  * Sequence design and evaluation
  * Structure prediction and evaluation

For each component, we extracted its function into a executable python file, which relies on manul input of sequences or structures. Each file could be directly executed in default setting as `python xxx.py`. For further usage, the detailed information is illustrated as following:


* Secondary structure generation
```shell
## This program aims to generate RNA secondary structure in dot-bracket format file and print it in the terminal. 
## The input should be the PATH to the PDB file and the specified chain name
conda activate CPD
cd R3Design/manul_input
python secondary_generation.py <'path to PDB file'> <'chain_name'>
## Example: 
## python secondary_generation.py '/root/a.pdb' 'A'
```

* Sequence design and evaluation

We also provided an implementation for sequence design in Colab, the model is trained on full RNAsolo dataset. Feel free to check out here:

<a href="https://colab.research.google.com/drive/1tAoUHY6w8WeweByY7TFwIyXPyGXA4lMW#scrollTo=gzAKWozrYdag" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```shell
## This program aims to generate RNA sequence using pre-trained R3Design model, based on the input of given PDB file. 
## Our R3Design model can only solve the single-chain RNA molecule, if the input PDB file has multiple chains, the system will use the first chain.
## The output would be the predicted sequence, the original sequence and evaluation metrics
conda activate CPD
cd R3Design/manul_input
python sequence_generation.py <'path to PDB file'>
## Example: 
## python sequence_generation.py '/root/a.pdb'
```

* Structure prediction and evaluation
```shell
## This program aims to predict the tertiary structure based on the fasta file given, and to evaluate the sturctural similarity through TM-Score and RMSD with other PDB files (If provided)

## The output would be the predicted PDB file and evaluation metrics(If Provided)
conda activate CPD
cd R3Design/manul_input
python structure_prediction.py <'path to fasta file'> <'path to PDB file to compare'>
## Example: 
## python sequence_generation.py '/root/a.fa' '/root/b.pdb'
```

## System Execution
Our system can perform sequence design from given PDB file, and execute structure prediction based on the designed sequence, providing predicted PDB file, and the evaluation metrics after comparing with the original PDB file.

```shell
## This program aims to generate RNA sequence using pre-trained R3Design model, based on the input of given PDB file. 
## Our R3Design model can only solve the single-chain RNA molecule, if the input PDB file has multiple chains, the system will use the first chain.
## The output would be written to the 'result' directory under the PDB file path, containing predicted sequence, PDB file, together with the metrics in the log.
conda activate CPD
cd R3Design/manul_input
python system_pipeline.py <'path to PDB file'>
## Example: 
## python sequence_generation.py '/root/a.pdb'
```
The logfile could be found at `R3Design/manul_input/full_result`. The logfile is `log.txt`, the predicted PDB file is inside the `RNA_name_pred` folder `eg: native_pred`.

##  Training from Scartch

Our tool could be trained from sctrach with simple commands. From the initial parameters, please refer to `parsers.py` in the `./R3Design` folder.
```shell
## For training, you could simply type: 
cd R3Design
python train.py 

```
## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.