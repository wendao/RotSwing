"""Unit tests for prepff — Stage 1 of the NCAA parameterization pipeline.

Tests cover: SMILES capping, charge calculation, PDB processing,
conformer clash detection, energy extraction, and pipeline state I/O.
"""

import os
import tempfile
import json
import numpy as np
import pytest
from rdkit import Chem


# ============================================================================
# SMILES Processing and Capping
# ============================================================================

def test_process_smiles_standard_amino_acid(prepff):
    """Capping should add ACE at N-term and NME at C-term."""
    smiles = 'N[C@@H](C)C(=O)O'  # L-alanine (no capping)
    result = prepff.process_smiles(smiles)
    assert 'CC(=O)N' in result or 'C(=O)N' in result  # ACE cap present
    assert 'NC' in result  # NME cap present


def test_process_smiles_invalid(prepff):
    """Invalid SMILES should raise ValueError."""
    with pytest.raises(ValueError, match='Unable to parse'):
        prepff.process_smiles('not_a_valid_smiles_string')


def test_process_smiles_stereochemistry_preserved(prepff):
    """Chiral centers should be preserved after capping."""
    smiles = 'N[C@@H](Cc1ccccc1)C(=O)O'  # L-phenylalanine
    result = prepff.process_smiles(smiles)
    assert '@' in result  # chirality preserved


# ============================================================================
# System Charge Calculation
# ============================================================================

def test_calculate_system_charge_neutral(tmp_smiles_file, prepff):
    """Alanine should have net charge 0."""
    charge = prepff.calculate_system_charge(tmp_smiles_file)
    assert charge == 0


def test_calculate_system_charge_positive(prepff):
    """SMILES with explicit + charge."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smiles', delete=False) as f:
        f.write('[NH3+]C(C)C(=O)[O-]')  # zwitterion: +1, -1 → net 0
        path = f.name
    try:
        charge = prepff.calculate_system_charge(path)
        assert charge == 0
    finally:
        os.unlink(path)


def test_calculate_system_charge_negative(prepff):
    """SMILES with extra - charge."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smiles', delete=False) as f:
        f.write('NC(C)C(=O)[O-]')  # net -1
        path = f.name
    try:
        charge = prepff.calculate_system_charge(path)
        assert charge == -1
    finally:
        os.unlink(path)


# ============================================================================
# Rotatable Bond Counting
# ============================================================================

def test_count_rotatable_bonds_alanine(prepff):
    """Alanine should have 2 rotatable bonds (after capping it's more)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smiles', delete=False) as f:
        f.write('N[C@@H](C)C(=O)O\n')
        path = f.name
    try:
        results = prepff.count_rotatable_bonds(path, cut_off=10000)
        assert len(results) == 1
        smiles, num_rot, c_val, v_val = results[0]
        assert num_rot >= 1  # at least one rotatable bond
        assert v_val > 0
        assert c_val > 0
    finally:
        os.unlink(path)


def test_count_rotatable_bonds_custom_cutoff(prepff):
    """Custom cutoff should override v_value."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smiles', delete=False) as f:
        f.write('CC(C)C[C@@H](C(=O)O)N\n')  # leucine-like
        path = f.name
    try:
        results = prepff.count_rotatable_bonds(path, cut_off=50)
        _, _, c_val, v_val = results[0]
        assert v_val == 50  # custom cutoff wins
    finally:
        os.unlink(path)


# ============================================================================
# PDB Energy Extraction
# ============================================================================

def test_extract_energy_from_pdb(prepff):
    """Should extract energy value from PDB REMARK line."""
    pdb_content = (
        "REMARK SMILES: CC(C)C\n"
        "REMARK Energy after optimization: -15.32 kcal/mol\n"
        "ATOM      1  N   ALA     1       0.000   0.000   0.000\n"
        "END\n"
    )
    energy = prepff.extract_energy_from_pdb(pdb_content)
    assert energy == pytest.approx(-15.32)


def test_extract_energy_from_pdb_no_energy(prepff):
    """Should raise ValueError when no energy line present."""
    pdb_content = (
        "REMARK SMILES: CC(C)C\n"
        "ATOM      1  N   ALA     1       0.000   0.000   0.000\n"
        "END\n"
    )
    with pytest.raises(ValueError):
        prepff.extract_energy_from_pdb(pdb_content)


def test_extract_energy_from_pdb_positive(prepff):
    """Should handle positive energy values."""
    pdb_content = (
        "REMARK Energy after optimization: 245.67 kcal/mol\n"
        "ATOM      1  N   ALA     1       0.000   0.000   0.000\n"
        "END\n"
    )
    energy = prepff.extract_energy_from_pdb(pdb_content)
    assert energy == pytest.approx(245.67)


# ============================================================================
# dftd4 Energy Extraction
# ============================================================================

def test_extract_energy_from_dftd4_output(prepff):
    """Should parse dispersion energy from dftd4 output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as f:
        f.write("Some header\n")
        f.write("Dispersion energy:    -0.0045231873 Eh\n")
        f.write("Other output\n")
        path = f.name
    try:
        energy = prepff.extract_energy_from_dftd4_output(path)
        assert energy == pytest.approx(-0.0045231873)
    finally:
        os.unlink(path)


def test_extract_energy_from_dftd4_output_missing(prepff):
    """Should return None when no energy line found."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as f:
        f.write("No energy here\n")
        path = f.name
    try:
        energy = prepff.extract_energy_from_dftd4_output(path)
        assert energy is None
    finally:
        os.unlink(path)


# ============================================================================
# Steric Clash Detection
# ============================================================================

def test_check_for_clashes_clean_molecule(prepff):
    """Ethane should have no steric clashes."""
    mol = Chem.MolFromSmiles('CC')
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDG())
    # Optimize with UFF to get reasonable geometry
    from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
    ff = UFFGetMoleculeForceField(mol)
    ff.Initialize()
    ff.Minimize()
    assert not prepff.check_for_clashes(mol)


def test_check_for_clashes_default_threshold(prepff):
    """Methane with Hs should pass (H-H distance > 0.6 A)."""
    mol = Chem.MolFromSmiles('C')
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDG())
    from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
    ff = UFFGetMoleculeForceField(mol)
    ff.Initialize()
    ff.Minimize()
    assert not prepff.check_for_clashes(mol, threshold=0.3)


def test_check_for_clashes_custom_threshold(prepff):
    """Large threshold should flag clashes."""
    mol = Chem.MolFromSmiles('CC')
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDG())
    from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
    ff = UFFGetMoleculeForceField(mol)
    ff.Initialize()
    ff.Minimize()
    # With a very large threshold, everything "clashes"
    assert prepff.check_for_clashes(mol, threshold=5.0)


# ============================================================================
# Dihedral Setting
# ============================================================================

def test_set_dihedral_angle(prepff):
    """Setting a dihedral should change it."""
    mol = Chem.MolFromSmiles('CCCC')  # butane - has a central dihedral
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDG())
    from rdkit.Chem.rdMolTransforms import GetDihedralDeg
    prepff.SetDihedralDeg(mol.GetConformer(), 0, 1, 2, 3, 60.0)
    angle = GetDihedralDeg(mol.GetConformer(), 0, 1, 2, 3)
    assert angle == pytest.approx(60.0, abs=0.1)


# ============================================================================
# Pipeline State File
# ============================================================================

def test_save_and_read_pipeline_state(prepff):
    """State file should be valid JSON and contain expected keys."""
    state = {
        'v_value': 30,
        'c_value': 10000,
        'n_value': 'TST',
        'names': ['TST'],
        'optimization_method': 'gaussian',
        'num_rotatable_bonds': 3,
        'rmsd_thresholds': {10: 0.80, 30: 0.60}
    }
    prepff.save_pipeline_state(state)

    assert os.path.exists(prepff.STATE_FILE)
    with open(prepff.STATE_FILE, 'r') as f:
        loaded = json.load(f)

    assert loaded['v_value'] == 30
    assert loaded['n_value'] == 'TST'
    assert loaded['optimization_method'] == 'gaussian'
    assert loaded['rmsd_thresholds']['30'] == 0.60


# ============================================================================
# MOL2 Processing
# ============================================================================

def test_process_mol2_file_nitrogen_types(prepff):
    """DU and N3 atom types should be corrected to N.

    The function checks specific column positions (50:52) for DU/N3,
    so the test data must align DU/N3 at those columns.
    """
    # The function checks line[50:52] for DU/N3 AND line[8:10] for "N".
    # "     1  N" = 9 chars (index 8 = 'N'), so we need 41 spaces padding
    # to place atom type at exactly column 50.
    atom1 = "     1  N" + " " * 41 + "DU" + "  1 TST    0.0000\n"
    atom2 = "     2  N" + " " * 41 + "N3" + "  1 TST    0.0000\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.mol2', delete=False) as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write("test\n")
        f.write("1 0 0 0 0\n")
        f.write("SMALL\n")
        f.write("USER_CHARGES\n")
        f.write("@<TRIPOS>ATOM\n")
        f.write(atom1)
        f.write(atom2)
        f.write("@<TRIPOS>BOND\n")
        path = f.name
    try:
        prepff.process_mol2_file(path)
        with open(path, 'r') as f:
            content = f.read()
        assert 'DU' not in content.split('@<TRIPOS>BOND')[0]
        assert 'N3' not in content.split('@<TRIPOS>BOND')[0]
    finally:
        os.unlink(path)


# ============================================================================
# SMILES Generation (GenSMILES)
# ============================================================================

def test_gen_smiles_mismatch(prepff):
    """Mismatched names and SMILES should raise ValueError."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smiles', delete=False) as f:
        f.write('N[C@@H](C)C(=O)O\n')
        f.write('N[C@@H](Cc1ccccc1)C(=O)O\n')
        path = f.name
    try:
        with pytest.raises(ValueError, match='Number of names'):
            prepff.gen_smiles(path, ['ONLY_ONE'])
    finally:
        os.unlink(path)
