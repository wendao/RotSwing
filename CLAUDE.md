# RotSwing - NCAA Parameterization System

## Project Overview

RotSwing is a modular pipeline for parameterizing **Non-Canonical Amino Acids (NCAAs)** for molecular modeling and simulation (Rosetta) and molecular dynamics (GROMACS). It bridges cheminformatics, quantum chemistry, and molecular mechanics.

## Quick Start

```bash
# Three-stage workflow for NCAA parameterization

# Stage 1: Generate intermediate files (SMILES → MOL2/PDB)
./scripts/prepff -i input.smiles -n XXX

# Stage 2: Generate Rosetta params files
./scripts/paramsgen -n XXX

# Stage 3: Generate GROMACS topology files
./scripts/topolgen -n XXX

# Or run everything in one command:
./scripts/rotswing -i input.smiles -n XXX
```

---

## Script Architecture

The pipeline consists of four scripts. Three stage scripts plus a master wrapper:

| Script | Stage | Input | Output |
|--------|-------|-------|--------|
| `prepff` | 1 | SMILES | MOL2, MOL, PDB |
| `paramsgen` | 2 | MOL, MOL2 | .params, rotamer PDB |
| `topolgen` | 3 | MOL2 | .top, .gro, .rtp |
| `rotswing` | all | SMILES | all of the above |

Stage scripts communicate via the filesystem. `prepff` writes a `.prepff_state.json` state file that `paramsgen` reads to get `v_value` (conformer count threshold) and other parameters.

**Deprecated scripts (deleted):** `Param_Rotamer.py`, `Param_Rotamer-after_opt.py`, `R_G_parameterize.py`. All their functionality has been absorbed into the modular pipeline above.

---

### 1. prepff - Prepare Force Field Intermediate Files

**Usage:**
```bash
prepff -i input.smiles -n XXX [-c cutoff] [-m gaussian|aimnet]
```

**Two optimization paths:**

#### Gaussian path (`-m gaussian`, default)
- SMILES → capping (ACE/NME) → conformer generation (RDKit ETKDG) → Gaussian DFT optimization (B3LYP/6-311+g(d,p)) → RESP charge fitting → MOL2 with RESP charges
- Produces high-accuracy charges but slow (hours)

#### AIMNet path (`-m aimnet`)
- SMILES → obabel 3D generation → UFF pre-optimization → AIMNet neural network optimization (gas phase 50 steps + solvent phase 499 steps, ASE BFGS with backbone dihedral constraints) → MOL2 with Gasteiger charges
- Fast (minutes), GPU required. No RESP charges — uses Gasteiger instead. Good for rapid screening.

**Conformer generation details:**
- Number of conformers (`c_value`, `v_value`) determined by rotatable bond count
- If dftd4 is available, each conformer gets dispersion energy calculated and stored in PDB REMARK
- Steric clash detection filters bad conformers (atoms < 0.6 Å apart)
- Conformers sorted by energy, first (lowest) used for QM optimization

**Output:**
- `RESP/XXX_1.mol2` — MOL2 with charges
- `mol/XXX_opt.mol` — MOL file with Rosetta polymer info
- `combined_sorted_pdb_files.pdb` — All conformers sorted by energy (input for paramsgen screening)
- `.prepff_state.json` — Pipeline state (v_value, c_value, etc.) for downstream scripts

---

### 2. paramsgen - Generate Rosetta Parameter Files

**Usage:**
```bash
paramsgen -n XXX [-r rmsd_threshold] [-e energy_cutoff] [-c]
```

**Screening pipeline (executed in order):**

1. **Energy screening** (`-e` flag): Discards conformers above `min_energy + cutoff` kcal/mol. Only works if PDB files contain energy data from `prepff` (requires dftd4).
2. **RMSD screening** (`-r` flag): Removes conformers with side-chain RMSD below threshold. Threshold auto-adapts based on `v_value` from prepff state: v=10→0.80Å, v=30→0.60Å, v=90→0.40Å, v=270→0.20Å, v=810→0.10Å. If user provides explicit threshold, that overrides the adaptive value.
3. **Chi angle screening**: Ensures diversity of side-chain dihedral angles (minimum 15° separation).
4. **Merge**: Combines RMSD-screened and Chi-screened sets, deduplicating by RMSD.

**After screening:**
- Generates Rosetta `.params` files via `molfile_to_params_polymer.py` (Python 2)
- Updates ICOOR parameters from MOL2 coordinates
- Adjusts charges to integer totals
- Adds `PDB_ROTAMERS` line referencing the rotamer library PDB

**Output:**
- `XXX.params` — Rosetta parameter file
- `XXX_temps.params` — Template (used for charge transfer)
- `merged_combined_pdb_files.pdb` — Rotamer library

---

### 3. topolgen - Generate GROMACS Topology Files

**Usage:**
```bash
topolgen -n XXX [--resp_folder RESP]
```

**Workflow:**
- `parmchk2` → missing parameter file (.mod)
- `tleap` → Amber topology (.prm, .crd)
- `acpype` → GROMACS conversion (.top, .gro)
- RTP generation from topology (excludes ACE/NME cap atoms, adds backbone connectivity bonds)

**Output:**
- `XXX_gromacs_prm/XXX.top` — GROMACS topology
- `XXX_gromacs_prm/XXX.gro` — GROMACS coordinates
- `XXX.rtp` — Residue topology file for pdb2gmx

---

### 4. rotswing - Master Pipeline Wrapper

**Usage:**
```bash
rotswing -i input.smiles -n XXX [options]

# Options from all stages:
#   -c CUT_OFF     Conformer generation cutoff (default: 10000)
#   -m METHOD      Optimization method: gaussian|aimnet (default: gaussian)
#   -r THRESHOLD   RMSD threshold (default: 0.001 = auto-adaptive)
#   -e CUTOFF      Energy cutoff in kcal/mol for dftd4 screening
#   --clean        Clean temporary files
#   --stage STAGE  Run single stage: prepff|paramsgen|topolgen|all
#   --dry-run      Print commands without executing
```

Runs `prepff → paramsgen → topolgen` in sequence, passing parameters through.

---

## Data Flow

```
input.smiles
     ↓
[prepff]
  ├─ Capping (ACE/NME)
  ├─ Conformer generation (RDKit + dftd4 energy)
  ├─ QM Optimization (Gaussian or AIMNet)
  ├─ RESP or Gasteiger charges → MOL2
  └─ MOL generation + polymer info
     ↓
.prepff_state.json  ←── v_value, c_value, etc.
     ↓
[paramsgen]
  ├─ Params file generation (molfile_to_params_polymer.py)
  ├─ ICOOR update
  ├─ Charge refinement
  ├─ Energy screening (optional, dftd4-based)
  ├─ RMSD screening (adaptive threshold)
  ├─ Chi angle screening
  └─ Rotamer library (.pdb)
     ↓
[topolgen]
  ├─ AmberTools (parmchk2, tleap)
  ├─ acpype (GROMACS conversion)
  └─ RTP generation
     ↓
Output: .params + .top + .rtp + rotamer PDB
```

---

## Pipeline State File

`prepff` writes `.prepff_state.json` with:

```json
{
  "v_value": 30,
  "c_value": 10000,
  "n_value": "03S",
  "names": ["03S"],
  "optimization_method": "gaussian",
  "num_rotatable_bonds": 3,
  "rmsd_thresholds": {"10": 0.80, "30": 0.60, "90": 0.40, "270": 0.20, "810": 0.10}
}
```

`paramsgen` reads this to get `v_value` for adaptive RMSD threshold selection.

---

## Technical Details

### Terminal Capping

ACE (acetyl, CH3-CO-) at N-terminus, NME (N-methyl amide, -NH-CH3) at C-terminus. Applied via SMARTS reaction transforms to the raw SMILES before any 3D operations.

### Conformer Generation Parameters

| Rotatable Bonds | v_value (target conformers) | c_value (max iterations) | Adaptive RMSD Threshold |
|-----------------|-----------------------------|--------------------------|-------------------------|
| 2 | 10 | 10,000 | 0.80 Å |
| 3 | 30 | 10,000 | 0.60 Å |
| 4 | 90 | 10,000 | 0.40 Å |
| 5 | 270 | 10,000 | 0.20 Å |
| 6+ | 810 | 10,000 | 0.10 Å |

### AIMNet Optimization Details

1. Initial 3D structure from obabel `--gen3D` (more stable than RDKit ETKDG for some scaffolds)
2. UFF force field pre-optimization
3. Backbone dihedral constraints: Phi=-150°, Psi=150° (or Phi=-120°, Psi=90° for peptoids)
4. Gas phase optimization: AIMNet-MT ensemble, BFGS, max 50 steps, fmax=0.001
5. Solvent phase optimization: AIMNet-SMD ensemble, BFGS, max 499 steps, fmax=0.001
6. PDB output with RDKit-compatible atom naming

### RESP Charge Calculation (Gaussian path only)

Two-stage restrained electrostatic potential fitting:
1. HF/6-31G* single point with MK population analysis
2. Two-stage RESP fit: unrestrained then hyperbolic restraint (0.001 a.u.)

AIMNet path uses Gasteiger charges via antechamber `-c gas` as a fast alternative.

### Cap Charge Redistribution

Total charges of ACE and NME cap groups are calculated and redistributed to the N-terminal N and C-terminal C atoms of the NCAA residue to maintain integer total charge.

### Screening Order Matters

Energy screening (optional) runs first and reduces the pool for subsequent RMSD screening. RMSD then Chi, and finally the two screened sets are merged. This ordering prioritizes thermodynamic accessibility (energy) over geometric diversity (RMSD) over rotamer diversity (Chi).

---

## Dependencies

- **Python 3.6+** with numpy, scipy, rdkit, biopython, pytest
- **Python 2.7** for Rosetta `molfile_to_params_polymer.py`
- **Gaussian 16** (for Gaussian path)
- **AIMNet + PyTorch + CUDA** (for AIMNet path)
- **AmberTools** (tleap, parmchk2, antechamber)
- **acpype** (Amber → GROMACS conversion)
- **Open Babel** (obabel, format conversion)
- **dftd4** (optional, for dispersion energy in conformer screening)
- **Rosetta** (optional, for validation)

---

## Testing

Tests are in `test/` and use pytest. The pipeline scripts (`prepff`, `paramsgen`, `topolgen`) have no `.py` extension, so `conftest.py` uses `importlib.machinery.SourceFileLoader` to load them as modules.

```bash
# Run all tests
python -m pytest test/ -v

# Run tests for a specific stage
python -m pytest test/test_prepff.py -v
python -m pytest test/test_paramsgen.py -v
python -m pytest test/test_topolgen.py -v
```

**Test coverage (62 tests total):**

| Test file | Tests | Coverage |
|-----------|-------|----------|
| `test/test_prepff.py` | 20 | SMILES processing, charge calculation, rotatable bonds, energy extraction, clash detection, dihedral setting, pipeline state I/O, MOL2 processing |
| `test/test_paramsgen.py` | 32 | Dihedral/angle geometry, Kabsch algorithm, RMSD, MOL2 reading, charge adjustment, conformer extraction, adaptive thresholds, chi comparison, energy/PBD parsing |
| `test/test_topolgen.py` | 10 | RTP generation: file creation, sections, cap exclusion, backbone bonds/impropers, atom renumbering, error handling |

Tests are designed to run without external dependencies (no Gaussian, AIMNet, AmberTools, or dftd4 required). They use temporary files and in-memory data.

---

## Common Issues

1. **SMILES parsing failure** → Validate with RDKit, check stereochemistry
2. **Gaussian convergence failure** → Check initial geometry, increase cycles
3. **AIMNet not available** → Falls back to Gaussian automatically
4. **dftd4 not found** → Conformer generation proceeds without energy annotation; energy screening in paramsgen will be skipped
5. **Clashing atoms** → Auto-detected and filtered (threshold: 0.6 Å)
6. **Charge imbalance** → Auto-corrected via `adjust_charges_to_integer()`
7. **Kabsch RMSD incorrect** → Fixed 2026-05-15: `kabsch_algorithm()` had a transpose error in the cross-covariance matrix (`H = P.T @ Q` changed to `H = Q.T @ P`) for correct row-vector convention. The old code returned the transpose of the optimal rotation; affected `calculate_rmsd()` for non-identical point sets.

---

## Performance

| Step | Gaussian Path | AIMNet Path |
|------|--------------|-------------|
| Conformer Gen (with dftd4) | 5-15 min | 5-15 min |
| Optimization | 2-6 hours | 5-15 min |
| Charge Fitting | 0.5-2 hours | < 1 min (Gasteiger) |
| **Total (1 residue)** | **2-6 hours** | **5-15 minutes** |

---

## Future Enhancements

- [ ] Explicit solvent support (SMD, PCM) in RESP calculation
- [ ] pKa prediction for titratable residues
- [ ] Integration with open-source QC packages (Psi4, xTB/GFN2-xTB)
- [ ] Support for D-amino acids, peptoids, beta/gamma amino acids
- [ ] Batch processing for libraries of NCAAs
- [ ] YAML/TOML configuration file support
