# RotSwing - Non-Canonical Amino Acid (NCAA) Parameterization Tools

This repository contains scripts for parameterizing non-natural amino acids (NCAAs) for use in Rosetta (molecular modeling) and GROMACS (molecular dynamics). The workflow supports generating rotamer libraries, RESP charges, and force field parameters.

---

## Table of Contents

- [Overview](#overview)
- [Main Scripts](#main-scripts)
  - [Param_Rotamer.py](#param_rotamerpy)
  - [Param_Rotamer-after_opt.py](#param_rotamer-after_optpy)
  - [R_G_parameterize.py](#r_g_parameterizepy)
- [Supporting Scripts](#supporting-scripts)
- [Utility Modules](#utility-modules)
- [Workflow Comparison](#workflow-comparison)
- [Usage Examples](#usage-examples)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)

---

## Overview

RotSwing provides a complete pipeline for NCAA parameterization:

1. **SMILES Input** → Parse non-natural amino acid structures
2. **Capping** → Add ACE (acetyl) and NME (N-methyl amide) terminal caps
3. **Conformer Generation** → Generate diverse 3D conformers using RDKit
4. **Optimization** → Optimize structures using quantum chemistry (Gaussian or AIMNet)
5. **Charge Calculation** → Calculate RESP (Restrained ElectroStatic Potential) charges
6. **Parameter Generation** → Create Rosetta .params files and GROMACS topology files

---

## Main Scripts

### Param_Rotamer.py

**Purpose**: Full-featured NCAA parameterization using **Gaussian** for quantum chemistry calculations.

**Key Features**:
- Generates multiple conformers based on rotatable bond count
- Filters conformers by RMSD (Root Mean Square Deviation) to ensure diversity
- Screens conformers by Chi angle diversity
- Performs Gaussian structure optimization with `B3LYP/6-311+g(d,p)` basis set
- Calculates RESP charges using Gaussian
- Generates Rosetta .params files via `molfile_to_params_polymer.py`
- Generates GROMACS topology files (.top, .rtp)
- Adjusts atom types and calculates cap charges automatically

**Main Functions**:

| Function | Description |
|----------|-------------|
| `count_rotatable_bonds(smiles_file, cut_off)` | Determines number of conformers based on rotatable bonds |
| `process_smiles(smiles)` | Adds ACE/NME caps to amino acid SMILES string |
| `smiles_to_pdb(smiles_list, names_list, output_dir)` | Converts SMILES to 3D PDB conformers using RDKit |
| `screen_rmsd(rmsd_threshold)` | Filters conformers by RMSD threshold |
| `Chi_screen(args_params)` | Screens conformers ensuring Chi angle diversity |
| `generate_opt(res, resp)` | Runs Gaussian optimization and RESP charge calculation |
| `atom_type_adjust(resp_folder)` | Adjusts atom types in mol2 files for consistency |
| `calculate_capcharge(resp_folder)` | Calculates and applies ACE/NME cap charges |
| `generate_top(resp_folder)` | Generates GROMACS topology (.top) file |
| `generate_rtp(resp_file)` | Generates GROMACS RTP (.rtp) file for residue library |

**Output Files**:
```
pdb_files/          - Generated 3D conformers (PDB format)
PDB_rearranged/     - Reordered PDB files for Gaussian
GJF/                - Gaussian input files (.gjf)
mol/                - MOL files for parameterization
log/                - Gaussian output logs
resp_out/           - RESP charge calculation outputs
params/             - Rosetta .params files
charge/             - Charge files
pdb2/               - Final processed PDB files
top/                - GROMACS topology files
```

---

### Param_Rotamer-after_opt.py

**Purpose**: Alternative parameterization workflow using different Gaussian settings (`B3LYP/LANL2DZ` basis set).

**Key Differences from Param_Rotamer.py**:
- Uses `B3LYP/LANL2DZ` instead of `B3LYP/6-311+g(d,p)`
- Simplified workflow with different optimization settings
- Suitable for systems where LANL2DZ pseudopotential is preferred

**Main Workflow**:
```python
def main(input_file, names_list, clean):
    # 1. Create directories: pdb_files, PDB_rearranged, GJF, mol
    # 2. Process SMILES → Generate conformers
    # 3. Optimize with Gaussian (B3LYP/LANL2DZ)
    # 4. Calculate RESP charges
    # 5. Generate params files
```

**Use When**:
- Working with systems containing heavy atoms
- When LANL2DZ pseudopotential is more appropriate
- As an alternative optimization approach

---

### R_G_parameterize.py

**Purpose**: Fast NCAA parameterization using **AIMNet** neural network instead of Gaussian.

**Key Features**:
- Uses AIMNet neural network for structure optimization (much faster than Gaussian)
- Uses dftd4 for dispersion energy calculations
- Generates rotamer libraries with energy-based screening
- Suitable for rapid parameterization of many compounds
- Less accurate than Gaussian but significantly faster

**Main Functions**:

| Function | Description |
|----------|-------------|
| `optimize_molecule(smiles, name)` | AIMNet-based geometry optimization |
| `smiles_to_pdb_rotamers(smiles_list, names_list, output_dir)` | Generates conformers with dftd4 energy calculation |
| `check_for_clashes(mol, threshold=0.6)` | Checks for atomic clashes in conformers |

**AIMNet Models**:
```python
model_gas = load_AIMNetMT_ens().cuda()   # Gas phase optimization
model_smd = load_AIMNetSMD_ens().cuda()  # Solvent phase optimization
```

**Use When**:
- Need rapid parameterization of many compounds
- High-throughput screening is required
- Some accuracy can be traded for speed
- GPU is available for neural network inference

---

## Supporting Scripts

### molfile_to_params_polymer.py

**Purpose**: Original Rosetta tool for converting MDL molfiles to Rosetta .params files.

**Key Functionality**:
- Parses MDL molfile format (.mol, .sdf)
- Generates Rosetta residue parameter files (.params)
- Handles polymer connectivity (backbone connections)
- Adds Rosetta-specific atom properties:
  - `ros_type`: Rosetta atom type
  - `mm_type`: Molecular mechanics atom type
  - `poly_upper`: Is upper connect atom (C-terminus)
  - `poly_lower`: Is lower connect atom (N-terminus)
  - `poly_n_bb`: Is backbone nitrogen
  - `poly_ca_bb`: Is backbone alpha carbon

### molfile_to_params_polymer_modify.py

**Purpose**: Modified/customized version of `molfile_to_params_polymer.py` with adaptations for RotSwing workflow.

**Modifications**:
- Customized for NCAA parameterization pipeline
- Enhanced handling of terminal caps (ACE/NME)
- Adjusted atom naming conventions
- Modified connectivity detection

---

## Utility Modules

Located in `scripts/python/rosetta_py/`:

### io/mdl_molfile.py
- MDL molfile parsing and writing
- Atom and bond handling
- 3D coordinate management

### utility/r3.py
- 3D vector operations
- Coordinate transformations
- RMSD calculations

### utility/rankorder.py
- Ranking and ordering utilities
- Sorting algorithms for conformers

---

## Workflow Comparison

| Feature | Param_Rotamer.py | Param_Rotamer-after_opt.py | R_G_parameterize.py |
|---------|------------------|---------------------------|---------------------|
| **QM Engine** | Gaussian | Gaussian | AIMNet |
| **Basis Set** | B3LYP/6-311+g(d,p) | B3LYP/LANL2DZ | Neural network |
| **Speed** | Slow (~hours) | Slow (~hours) | Fast (~minutes) |
| **Accuracy** | High | High | Moderate |
| **GPU Required** | No | No | Yes (recommended) |
| **Conformer Gen** | RDKit + RMSD filter | RDKit | RDKit + dftd4 |
| **Best For** | Production accuracy | Heavy atom systems | High-throughput |

---

## Usage Examples

### Basic Usage - Param_Rotamer.py

```bash
cd /path/to/RotSwing

# Prepare input files
# 1. SMILES file (e.g., compounds.smi):
# CC(C)C[C@H](NC(=O)C)C(=O)NC    # Leucine analog
# ...

# 2. Names file (e.g., names.txt):
# LEU_ANALOG_01
# ...

# Run parameterization
python scripts/Param_Rotamer.py compounds.smi names.txt
```

### Basic Usage - R_G_parameterize.py

```bash
# Ensure GPU is available
python scripts/R_G_parameterize.py compounds.smi names.txt
```

### Input File Format

**SMILES file** (one per line):
```
CC(C)C[C@H](N)C(=O)O      # Natural amino acid
C1=CC=C(C=C1)C[C@H](N)C(=O)O    # Phenylalanine
```

**Names file** (one per line, matching SMILES order):
```
LEU
PHE
```

---

## Directory Structure

```
RotSwing/
├── scripts/
│   ├── Param_Rotamer.py                    # Main Gaussian-based workflow
│   ├── Param_Rotamer-after_opt.py          # Alternative Gaussian workflow
│   ├── R_G_parameterize.py                 # AIMNet-based fast workflow
│   ├── molfile_to_params_polymer.py        # Rosetta params generator
│   ├── molfile_to_params_polymer_modify.py # Modified params generator
│   └── python/
│       └── rosetta_py/
│           ├── __init__.py
│           ├── io/
│           │   ├── __init__.py
│           │   └── mdl_molfile.py          # Molfile I/O
│           └── utility/
│               ├── __init__.py
│               ├── r3.py                   # 3D vectors
│               └── rankorder.py            # Ranking utilities
├── examples/                               # Example input/output files
└── README.md                               # This file
```

---

## Dependencies

### Required Python Packages

```bash
# Core dependencies
numpy
rdkit

# For Gaussian workflows
# - Gaussian 09 or 16 (must be installed and in PATH)

# For AIMNet workflow
aimnet
dftd4
torch

# For GROMACS topology generation
# - GROMACS (must be installed and in PATH)

# For Rosetta parameterization
# - Rosetta (optional, for validation)
```

### Installation

```bash
# Install Python dependencies
pip install numpy rdkit torch

# Install AIMNet (for R_G_parameterize.py)
# Follow AIMNet installation instructions

# Install dftd4
# https://github.com/dftd4/dftd4

# Ensure Gaussian is available in PATH
# export g16root=/path/to/gaussian
# source $g16root/g16/bsd/g16.profile
```

---

## Output File Descriptions

### Rosetta .params file
Contains residue topology, atom types, bond connectivity, and icoordinate data for use in Rosetta simulations.

### GROMACS .top file
Topology file containing atom types, charges, bonds, angles, and dihedrals for molecular dynamics simulations.

### GROMACS .rtp file
Residue topology entry for the GROMACS residue database, enabling the custom residue to be used in GROMACS workflows.

### Rotamer Library
Generated conformers with diverse Chi angles for use in Rosetta rotamer libraries, enabling accurate side-chain sampling.

---

## References

- **Rosetta**: https://www.rosettacommons.org/
- **RDKit**: https://www.rdkit.org/
- **Gaussian**: https://gaussian.com/
- **AIMNet**: https://github.com/isayevlab/AIMNet
- **RESP**: Bayly, C.I. et al. (1993). J. Phys. Chem. 97, 10269-10280
- **GROMACS**: https://www.gromacs.org/

---

## License

See repository LICENSE file for details.

## Citation

If you use these tools in your research, please cite:
- Original Rosetta tools and methods
- The respective quantum chemistry packages used (Gaussian or AIMNet)
