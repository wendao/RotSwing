# RotSwing

A modular pipeline for parameterizing **Non-Canonical Amino Acids (NCAAs)** for Rosetta molecular modeling and GROMACS molecular dynamics simulations.

## Architecture

The pipeline consists of three stages plus a master wrapper:

| Script | Stage | Input | Output |
|--------|-------|-------|--------|
| `prepff` | 1 | SMILES | MOL2, MOL, PDB |
| `paramsgen` | 2 | MOL, MOL2 | .params, rotamer PDB |
| `topolgen` | 3 | MOL2 | .top, .gro, .rtp |
| `rotswing` | all | SMILES | all of the above |

## Dependencies

Tested under Python 3.8.18. Python 2.7.17 is required for Rosetta's `molfile_to_params_polymer.py`.

Core Python packages:
- numpy, scipy, rdkit, biopython, pytest

Optional (path-dependent):
- **Gaussian 16** — DFT optimization and RESP charges (Gaussian path)
- **AIMNet + PyTorch + CUDA** — neural network optimization (AIMNet path, fast)
- **AmberTools + acpype** — GROMACS topology generation (Stage 3)
- **Open Babel** — format conversion, 3D structure generation
- **dftd4** — dispersion energy for conformer screening

```bash
# Recommended install
conda install -c conda-forge openbabel ambertools=22 acpype dftd4
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Quick Start

```bash
# Three-stage workflow for a single NCAA

# Stage 1: Generate intermediate files (SMILES → MOL2/PDB)
./scripts/prepff -i input.smiles -n XXX

# Stage 2: Generate Rosetta params files
./scripts/paramsgen -n XXX

# Stage 3: Generate GROMACS topology files
./scripts/topolgen -n XXX

# Or run everything in one command:
./scripts/rotswing -i input.smiles -n XXX
```

## Input

An `input.smiles` file containing one SMILES string per NCAA:

```
N[C@H](C(O)=O)CC1=C(C(F)=C(C(F)=C1F)F)F
```

Multiple SMILES can be processed at once (separated by newlines), but the number of NCAA three-letter codes (`-n`) must match.

## Options

### prepff (Stage 1)
| Flag | Description | Default |
|------|-------------|---------|
| `-i` | Input SMILES file | required |
| `-n` | Three-letter NCAA code | required |
| `-c` | Conformer generation cutoff | 10000 |
| `-m` | Optimization method: `gaussian` or `aimnet` | gaussian |

### paramsgen (Stage 2)
| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Three-letter NCAA code | required |
| `-r` | RMSD threshold (0.001 = auto-adaptive) | 0.001 |
| `-e` | Energy cutoff in kcal/mol for dftd4 screening | none |
| `-c` | Clean temporary files | false |

### topolgen (Stage 3)
| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Three-letter NCAA code | required |
| `--resp_folder` | Custom RESP output folder | RESP |

### rotswing (all stages)
Accepts all options above plus:
| Flag | Description |
|------|-------------|
| `--stage STAGE` | Run single stage: `prepff`, `paramsgen`, `topolgen`, or `all` |
| `--dry-run` | Print commands without executing |
| `--clean` | Clean temporary files |

## Two Optimization Paths

### Gaussian Path (`-m gaussian`, default)
High accuracy, slow (2-6 hours). DFT optimization at B3LYP/6-311+g(d,p) level with RESP charge fitting.

### AIMNet Path (`-m aimnet`)
Fast (5-15 minutes), requires GPU. Neural network optimization with Gasteiger charges. Good for rapid screening.

| Step | Gaussian | AIMNet |
|------|----------|--------|
| Conformer Gen | 5-15 min | 5-15 min |
| Optimization | 2-6 hours | 5-15 min |
| Charge Fitting | 0.5-2 hours | < 1 min |
| **Total** | **2-6 hours** | **5-15 min** |

## Output

- `XXX.params` — Rosetta parameter file
- `XXX.rtp` — GROMACS residue topology for pdb2gmx
- `XXX_gromacs_prm/XXX.top` — GROMACS topology
- `XXX_gromacs_prm/XXX.gro` — GROMACS coordinates
- `merged_combined_pdb_files.pdb` — Rotamer library

## Testing

```bash
# Run all 62 tests
python -m pytest test/ -v

# Run per stage
python -m pytest test/test_prepff.py -v     # 20 tests
python -m pytest test/test_paramsgen.py -v  # 32 tests
python -m pytest test/test_topolgen.py -v   # 10 tests
```

Tests run without external dependencies (no Gaussian, AIMNet, AmberTools, or dftd4 needed).

## References

- [CLAUDE.md](CLAUDE.md) — Detailed technical documentation
- [ALGORITHM.md](ALGORITHM.md) — Algorithm descriptions
- [USAGE.md](USAGE.md) — Extended usage guide
