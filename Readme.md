# RotSwing
A protein modeling and sampling method for non canonical amino acids, post-translational modifications, and covalent modifications

## Dependence
This script is tested under Python 3.8.18.
Also, Python 2.7.17 is required to run Rosetta related scripts.
Other software and libraries that need to be installed:
- numpy 1.24.4
- scipy 1.10.1
- rdkit 2024.03.5
- biopython 1.83
- aimnet
- torch 2.3.1
- cudatoolkit 11.3.1
- ase 3.23.0
- dftd4 3.6.0
- dftd4-python 3.6.0

## Input
An **input.smiles** file which contains SMILES format test of your molecule is required, the content is similar to the following form:

`N[C@H](C(O)=O)CC1=C(C(F)=C(C(F)=C1F)F)F`

Note: If you want to process multiple SMILES at once, you can separate them with line breaks. And there should be no empty lines at the beginning, end, or middle of the file. At the same time, you need to ensure that the number of NCAA three letter abbreviations entered in **-n** command matches the number of SMILES that are entered. Although this is feasible, it is **<font color=red>not recommended</font>** to add multiple smiles at once in input.smiles, as this may cause errors during runtime
## Run
In order to generate Rosetta and Gromacs parameters, you can refer to the following steps:
1. Create a specific folder for your NCAA
2. Place the **script** and **input.smiles** file into this folder
3. Enter the following command to run the script
```
cd ~/YOUR_NCAA_FOLDER
python Param_Rotamer.py -i input.smiles -n UAA
```
Required:
-i: your NCAA's SMILES
-n: your NCAA's three letter abbreviation

Optional:
-c: whether or not delete intermediate files, accept 0 or 1 as its parameter. 0 stands for 'do not delete', 1 stands for 'delete'. Default delete
-d: RMSD_threshold, which is used to determine which conformations will be retained. It is recommended to choose a value between **0.1 and 1**. A larger RMSD_threshold will result in greater differences between conformations in the library, higher sparsity, and fewer conformations. 
## Output
You will find your Rosetta parameter file **UAA.params**, Rotamer library file **merged_combined_pdb_files.pdb**, and Gromacs parameter files in the current folder
