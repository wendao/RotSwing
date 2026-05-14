# RotSwing Usage Guide

## New Modular Architecture

The `Param_Rotamer.py` script has been refactored into three independent scripts:

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `prepff` | Prepare intermediate files | SMILES | MOL2, MOL, PDB |
| `paramsgen` | Generate Rosetta params | MOL/MOL2 | .params, rotamer PDB |
| `topolgen` | Generate GROMACS topology | MOL2 | .top, .gro, .rtp |

---

## Installation

Add scripts to your PATH:

```bash
export PATH=$PATH:/path/to/RotSwing/scripts
```

Or use directly:

```bash
./scripts/prepff -i input.smiles -n XXX
```

---

## Quick Start

### Option 1: Run Full Pipeline (One Command)

```bash
rotswing -i examples/input.smiles -n 03S
```

### Option 2: Run Stages Separately

```bash
# Stage 1: Generate intermediate files
prepff -i input.smiles -n 03S

# Stage 2: Generate Rosetta params
paramsgen -n 03S

# Stage 3: Generate GROMACS topology
topolgen -n 03S
```

---

## Detailed Usage

### prepff

Generate intermediate files from SMILES input.

```bash
prepff -i input.smiles -n XXX [options]

Options:
  -i, --input         Input SMILES file (required)
  -n, --names         Residue name(s) (required)
  -c, --cut_off       Conformer generation cutoff (default: 10000)
  -m, --method        Optimization method: gaussian|aimnet (default: gaussian)
```

**Example:**
```bash
prepff -i input.smiles -n 03S -c 50 -m gaussian
```

**Output:**
- `RESP/03S_1.mol2` - MOL2 file with RESP charges
- `mol/03S_opt.mol` - MOL file for Rosetta
- `PDB_rearranged/` - Directory with processed PDB files

---

### paramsgen

Generate Rosetta parameter files from intermediate files.

```bash
paramsgen -n XXX [options]

Options:
  -n, --name          Residue name (required)
  -r, --rmsd_threshold  RMSD threshold for screening (default: 0.001)
  -c, --clean         Clean temporary files after completion
  --resp_folder       Path to RESP folder (default: RESP)
```

**Example:**
```bash
paramsgen -n 03S -r 0.5 -c
```

**Output:**
- `03S.params` - Rosetta parameter file
- `03S_temps.params` - Template params file
- `merged_combined_pdb_files.pdb` - Rotamer library PDB

---

### topolgen

Generate GROMACS topology files from MOL2.

```bash
topolgen -n XXX [options]

Options:
  -n, --name          Residue name (required)
  --resp_folder       Path to RESP folder (default: RESP)
```

**Example:**
```bash
topolgen -n 03S
```

**Output:**
- `03S_gromacs_prm/03S.top` - GROMACS topology
- `03S_gromacs_prm/03S.gro` - GROMACS coordinates
- `03S.rtp` - Residue topology file

---

## rotswing (Master Script)

Convenience wrapper to run all stages.

```bash
rotswing -i input.smiles -n XXX [options]

Options:
  -i, --input         Input SMILES file (required)
  -n, --names         Residue name(s) (required)
  -c, --cut_off       Conformer cutoff (default: 10000)
  -m, --method        Optimization method (default: gaussian)
  -r, --rmsd_threshold  RMSD threshold (default: 0.001)
  --clean             Clean temporary files
  --stage             Run specific stage: prepff|paramsgen|topolgen|all
  --dry-run           Print commands without executing
```

**Examples:**

```bash
# Full pipeline
rotswing -i input.smiles -n 03S

# Run only stage 1
rotswing -i input.smiles -n 03S --stage prepff

# Dry run (see what would be executed)
rotswing -i input.smiles -n 03S --dry-run

# Custom settings
rotswing -i input.smiles -n 03S -c 50 -r 0.5 --clean
```

---

## Directory Structure After Running

```
.
├── input.smiles              # Input file
├── RESP/
│   └── 03S_1.mol2           # MOL2 with RESP charges
├── mol/
│   └── 03S_opt.mol          # MOL file for Rosetta
├── PDB_rearranged/          # Processed PDB files
├── GJF/                     # Gaussian input files
├── 03S.params               # Rosetta parameter file
├── 03S_temps.params         # Template params file
├── 03S.rtp                  # GROMACS residue topology
├── 03S_gromacs_prm/         # GROMACS files
│   ├── 03S.top
│   ├── 03S.gro
│   └── ...
└── merged_combined_pdb_files.pdb  # Rotamer library
```

---

## Migration from Param_Rotamer.py

If you were using the old `Param_Rotamer.py`:

| Old Command | New Command |
|-------------|-------------|
| `python Param_Rotamer.py -i input.smiles -n XXX` | `rotswing -i input.smiles -n XXX` |
| `python Param_Rotamer.py -i input.smiles -n XXX -d 0.5` | `rotswing -i input.smiles -n XXX -r 0.5` |
| `python Param_Rotamer.py -i input.smiles -n XXX -f 50` | `rotswing -i input.smiles -n XXX -c 50` |

---

## Requirements

- Python 3.6+
- RDKit
- AmberTools (tleap, parmchk2)
- Gaussian 16 (or AIMNet)
- Open Babel
- acpype
- Rosetta (molfile_to_params_polymer.py)
- BioPython
- NumPy
