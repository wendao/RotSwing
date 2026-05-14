# RotSwing - NCAA Parameterization System

## Project Overview

RotSwing is a comprehensive system for parameterizing **Non-Canonical Amino Acids (NCAAs)** for molecular modeling and simulation. It bridges cheminformatics, quantum chemistry, and molecular mechanics to generate production-ready force field parameters.

## Quick Start

```bash
# Three-stage workflow for NCAA parameterization

# Stage 1: Generate intermediate files (SMILES → MOL2/PDB)
./scripts/prepff -i input.smiles -n XXX

# Stage 2: Generate Rosetta params files
./scripts/paramsgen -n XXX

# Stage 3: Generate GROMACS topology files
./scripts/topolgen -n XXX
```

---

## Script Architecture

The parameterization pipeline is split into three modular scripts:

### 1. prepff - Prepare Force Field Intermediate Files

**Purpose**: Generate intermediate files from input SMILES

**Functions**:
- SMILES parsing and validation
- Terminal capping (ACE/NME)
- Conformer generation (RDKit ETKDG)
- Structure optimization (Gaussian/AIMNet)
- RESP charge calculation
- Atom type assignment
- Cap charge redistribution

**Input**:
- `input.smiles` - SMILES string(s)

**Output**:
- `RESP/XXX_1.mol2` - MOL2 with RESP charges
- `mol/XXX_opt.mol` - MOL file for Rosetta
- `PDB_rearranged/` - Processed PDB files

**Usage**:
```bash
prepff -i input.smiles -n XXX [-c cutoff] [-m gaussian|aimnet]
```

---

### 2. paramsgen - Generate Rosetta Parameter Files

**Purpose**: Create `.params` files for Rosetta from intermediate files

**Functions**:
- Convert MOL to params (molfile_to_params_polymer.py)
- ICOOR parameter calculation and update
- Charge refinement (round to integers)
- RMSD-based conformer screening
- Chi angle diversity screening
- Rotamer library PDB generation

**Input**:
- `mol/XXX_opt.mol` - from prepff
- `RESP/XXX_1.mol2` - from prepff

**Output**:
- `XXX.params` - Rosetta parameter file
- `XXX_temps.params` - Template params (for charge transfer)
- `merged_combined_pdb_files.pdb` - Rotamer library

**Usage**:
```bash
paramsgen -n XXX [-r rmsd_threshold] [-c]
```

---

### 3. topolgen - Generate GROMACS Topology Files

**Purpose**: Create `.top` and `.rtp` files for GROMACS

**Functions**:
- AmberTools parmchk2 for missing parameters
- tleap for Amber topology generation
- acpype for GROMACS conversion
- RTP file generation (residue template)

**Input**:
- `RESP/XXX_1.mol2` - from prepff

**Output**:
- `XXX_gromacs_prm/XXX.top` - GROMACS topology
- `XXX_gromacs_prm/XXX.gro` - GROMACS coordinates
- `XXX.rtp` - Residue topology file

**Usage**:
```bash
topolgen -n XXX [--resp_folder RESP]
```

---

## Data Flow

```
input.smiles
     ↓
[prepff] → ACE/XXX/NME capping
     ↓
Conformer generation (RDKit)
     ↓
QM Optimization (Gaussian/AIMNet)
     ↓
RESP Charge fitting
     ↓
┌─────────────┬─────────────┐
↓             ↓             ↓
[paramsgen]  [topolgen]    (intermediate)
     ↓             ↓
XXX.params   XXX_gromacs_prm/
(Rosetta)    XXX.top
             XXX.rtp
             (GROMACS)
```

---

### 1. Input Processing

**Purpose**: Accept and validate non-natural amino acid descriptions

**Requirements**:
- Parse SMILES strings representing amino acid structures
- Support standard and non-standard amino acid backbones
- Validate chemical structure correctness
- Handle stereochemistry (R/S, D/L configurations)

**Inputs**:
- SMILES notation (e.g., `CC(C)C[C@H](N)C(=O)O`)
- Residue names/identifiers
- Optional: existing 3D structures (PDB, MOL)

**Outputs**:
- Validated molecular structure objects
- Standardized atom naming conventions

---

### 2. Terminal Capping

**Purpose**: Block amino acid termini to create realistic peptide environment

**Requirements**:
- Automatically add ACE (acetyl, CH3-CO-) cap at N-terminus
- Automatically add NME (N-methyl amide, -NH-CH3) cap at C-terminus
- Maintain proper bond connectivity and geometry
- Preserve stereochemistry during capping

**Why Capping Matters**:
- Prevents artificial charge effects at termini
- Mimics actual peptide chain environment
- Required for accurate RESP charge calculation
- Standard practice in force field development

---

### 3. Conformer Generation

**Purpose**: Generate diverse 3D conformations for side-chain flexibility

**Requirements**:
- Generate multiple conformers based on rotatable bond count
- Use systematic approach to sample chi angles (side-chain dihedrals)
- Ensure conformer diversity (geometric coverage)
- Avoid atomic clashes (steric hindrance)

**Algorithms**:
- Distance geometry (RDKit ETKDG)
- Random/ systematic dihedral variation
- RMSD-based diversity filtering
- Energy-based filtering (DFTD4 for dispersion correction)

**Configuration**:
```
Rotatable Bonds  →  Min Conformers
1-2              →  50
3-4              →  100
5+               →  200
```

---

### 4. Structure Optimization

**Purpose**: Refine 3D structures to energy minima

**Two Implementation Paths**:

#### Path A: Quantum Chemistry (Gaussian)
- **Method**: DFT (Density Functional Theory)
- **Functionals**: B3LYP
- **Basis Sets**: 6-311+g(d,p) or LANL2DZ (for heavy atoms)
- **Settings**: Tight convergence, ultrafine integration grid
- **Pros**: High accuracy, established method
- **Cons**: Computationally expensive (hours per molecule)

#### Path B: Neural Network (AIMNet)
- **Method**: Machine learning potential
- **Models**: AIMNet-MT (gas phase), AIMNet-SMD (solution phase)
- **Acceleration**: GPU (CUDA)
- **Pros**: Fast (minutes), reasonable accuracy
- **Cons**: Less accurate than DFT, GPU-dependent

**Optimization Criteria**:
- Force convergence < 0.0001 Hartree/Bohr
- Displacement convergence < 0.0001 Bohr
- Maximum iterations: 100-200

---

### 5. Conformer Screening

**Purpose**: Select representative conformers for parameterization

**Multi-Level Screening**:

1. **RMSD Screening** (Geometric Diversity)
   - Remove conformers with RMSD < threshold to existing set
   - Default threshold: 0.5-1.0 Å
   - Keeps geometrically distinct conformations

2. **Chi Angle Screening** (Rotamer Diversity)
   - Calculate all side-chain dihedral angles (chi1, chi2, chi3, chi4)
   - Bin conformers by chi angle regions
   - Ensure coverage of all rotameric states (gauche+, gauche-, trans)
   - Minimum angular separation: 30°

3. **Energy Screening** (Thermodynamic Feasibility)
   - Calculate single-point energies
   - Discard high-energy conformers (>10 kcal/mol above minimum)
   - Boltzmann weighting for population analysis

---

### 6. RESP Charge Calculation

**Purpose**: Derive accurate atomic partial charges

**Methodology**:
- **RESP**: Restrained ElectroStatic Potential
- Two-stage fitting:
  1. Unrestrained fit to ESP (Electrostatic Potential)
  2. Restrained fit with hyperbolic restraint (0.001 a.u.)

**Procedure**:
1. Calculate electrostatic potential grid (Gaussian)
2. Fit charges to reproduce ESP at grid points
3. Apply restraints to reduce overfitting
4. Constrain cap atoms to standard values
5. Scale charges to reproduce molecular dipole

**Output**:
- Partial atomic charges (RESP2 for aliphatic, RESP1 for polar)
- Charge equivalence groups for chemically equivalent atoms

---

### 7. Atom Typing

**Purpose**: Assign force field atom types for molecular mechanics

**Rosetta Atom Types**:
- Backbone atoms: Nbb, CAbb, Cbb, OCbb
- Side-chain types: based on chemical environment
- Special types for aromatic, polar, charged groups

**GAFF/AMBER Atom Types** (for GROMACS):
- sp3 carbons (c3)
- sp2 carbons (c2, ca for aromatic)
- Oxygens (oh, o)
- Nitrogens (n, n3, n4)
- Hydrogens (h1, ha, hn)

**Rules**:
- Hybridization state
- Number of attached atoms
- Chemical environment (aromatic, polar, etc.)
- Bond order patterns

---

### 8. Force Field Parameter Generation

#### 8.1 Rosetta Parameters (.params file)

**Contents**:
- Atom records: name, Rosetta type, MM type, charge, coordinates
- Bond connectivity: ICOOR (internal coordinate) records
- Properties: LJ radii, LJ depth, acceptor/donor flags
- Polymer info: lower/upper connect atoms, NBR (neighbor) atom
- Chi definitions: rotatable bonds

**Special Handling**:
- Backbone connectivity (N-CA-C)
- Side-chain torsions (chi angles)
- Terminal capping groups
- Aromatic rings and planarity constraints

#### 8.2 GROMACS Topology (.top file)

**Sections**:
- [ defaults ]: Force field defaults
- [ atomtypes ]: Atom type definitions
- [ moleculetype ]: Residue name and exclusions
- [ atoms ]: Atom numbers, types, charges, masses
- [ bonds ]: Bond parameters (lengths, force constants)
- [ angles ]: Angle parameters
- [ dihedrals ]: Proper and improper dihedral parameters

#### 8.3 GROMACS RTP (.rtp file)

**Purpose**: Residue template for pdb2gmx

**Sections**:
- [ bondedtypes ]: Bond/angle/dihedral type mapping
- Residue entry: atoms, bonds, angles, dihedrals, impropers

---

### 9. Cap Charge Adjustment

**Purpose**: Neutralize terminal cap contributions

**Procedure**:
1. Calculate total charge of ACE and NME caps separately
2. Distribute cap charge excess across main residue atoms
3. Maintain integer total charge (usually 0 for zwitterion, ±1 for charged)
4. Preserve chemical symmetry in charge redistribution

**ACE Standard Charges**:
- C (carbonyl): +0.6163
- O (carbonyl): -0.5722
- CH3 methyl: -0.3662 (distributed)

**NME Standard Charges**:
- N: -0.4632
- H: +0.2772
- C (carbonyl): +0.6163
- O (carbonyl): -0.5722
- CH3 methyl: -0.3662 (distributed)

---

### 10. Validation and Quality Control

**Checks**:
- Total charge equals expected value (±0.01 e tolerance)
- No atomic clashes (minimum distance > 1.2 Å)
- Bond lengths within chemical norms
- Improper dihedrals maintain chirality
- Ring closure distances acceptable

**Output Metrics**:
- Number of rotatable bonds
- Number of generated conformers
- Number of screened conformers
- RMSD between conformers
- Energy spread (kcal/mol)

---

## Data Flow Architecture

```
Input SMILES
     ↓
[Capping] → ACE + NCAA + NME
     ↓
[Conformer Generation] → N conformers
     ↓
[Optimization] → Gaussian OR AIMNet
     ↓
[Conformer Screening] → RMSD + Chi + Energy filters
     ↓
[RESP Calculation] → Atomic charges
     ↓
[Atom Typing] → Rosetta + GAFF types
     ↓
[Parameter Generation] → .params + .top + .rtp
     ↓
Output Files
```

---

## Configuration Parameters

### Conformer Generation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `rmsd_threshold` | 0.5 Å | Minimum RMSD between conformers |
| `chi_separation` | 30° | Minimum chi angle difference |
| `max_conformers` | 200 | Upper limit for conformer count |
| `energy_cutoff` | 10 kcal/mol | Maximum energy above minimum |

### Gaussian Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `functional` | B3LYP | DFT exchange-correlation functional |
| `basis_set` | 6-311+g(d,p) | Gaussian basis set |
| `solvent` | None | Solvent model (SMD, PCM, etc.) |
| `maxcyc` | 200 | Maximum optimization cycles |

### AIMNet Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | cuda | Computation device |
| `fmax` | 0.001 | Force convergence threshold |
| `steps` | 500 | Maximum optimization steps |

---

## Use Cases

### Use Case 1: Drug Design
**Scenario**: Incorporate unnatural amino acids into peptide therapeutics
**Workflow**: Param_Rotamer.py with Gaussian for accurate charges
**Output**: Rosetta params for docking/design, GROMACS topology for MD

### Use Case 2: Enzyme Engineering
**Scenario**: Model enzyme active sites with modified residues
**Workflow**: R_G_parameterize.py for rapid screening of many variants
**Output**: Fast parameterization for virtual screening

### Use Case 3: Force Field Development
**Scenario**: Build custom force field for specific chemical space
**Workflow**: Param_Rotamer.py with careful RESP fitting
**Output**: Validated parameters with complete documentation

---

## Integration Points

### External Software
- **RDKit**: Cheminformatics toolkit (SMILES, conformers)
- **Gaussian**: Quantum chemistry engine
- **AIMNet**: Neural network potential
- **Rosetta**: Protein modeling suite
- **GROMACS**: Molecular dynamics engine
- **DFTD4**: Dispersion correction

### File Formats
- **Input**: SMILES, SDF, MOL, PDB
- **Intermediate**: XYZ, GJF, LOG, MOL2
- **Output**: PARAMS, TOP, RTP, PDB

---

## Error Handling

### Common Issues
1. **SMILES parsing failure** → Validate with RDKit, check stereochemistry
2. **Gaussian convergence failure** → Check initial geometry, increase cycles
3. **Clashing atoms** → Increase conformer generation, lower RMSD threshold
4. **Charge imbalance** → Verify cap charge calculation, check atom typing
5. **Missing parameters** → Add custom bond/angle parameters

### Recovery Strategies
- Fallback to different initial conformers
- Manual geometry adjustment
- Custom atom type assignment
- Multi-step optimization (coarse → fine)

---

## Performance Considerations

| Step | Gaussian Path | AIMNet Path |
|------|--------------|-------------|
| Conformer Gen | Minutes | Minutes |
| Optimization | Hours | Minutes |
| RESP Fitting | Hours | N/A |
| Total (1 residue) | 2-6 hours | 5-15 minutes |

**Parallelization**:
- Conformer generation: Embarrassingly parallel
- Gaussian jobs: Can be distributed across cluster
- AIMNet: GPU batch processing

---

## Future Enhancements

- [ ] Explicit solvent support (SMD, PCM)
- [ ] pKa prediction for titratable residues
- [ ] Automated validation against QM benchmarks
- [ ] Integration with open-source QC packages (Psi4, xTB)
- [ ] Web interface for non-expert users
- [ ] Batch processing for libraries of NCAAs
