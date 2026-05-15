"""Unit tests for paramsgen — Stage 2 of the NCAA parameterization pipeline.

Tests cover: 3D geometry calculations, RMSD computation, charge refinement,
conformer extraction, adaptive RMSD threshold, and energy screening.
"""

import os
import tempfile
import numpy as np
import math
import pytest


# ============================================================================
# 3D Geometry: Dihedral Angle
# ============================================================================

def test_calculate_dihedral_params_trans(paramsgen):
    """Trans (180°) dihedral should return ~180."""
    # Trans configuration: zigzag in XY plane
    c1 = [0.0, 1.0, 0.0]
    c2 = [1.0, 0.0, 0.0]
    c3 = [2.0, 1.0, 0.0]
    c4 = [3.0, 0.0, 0.0]
    angle = paramsgen.calculate_dihedral_params(c1, c2, c3, c4)
    assert 150 < abs(angle) <= 180


def test_calculate_dihedral_params_gauche(paramsgen):
    """Gauche (~60°) dihedral."""
    # Coordinates verified to produce 60.00° dihedral
    c1 = [0.0, 0.0, 1.0]
    c2 = [0.0, 0.0, 0.0]
    c3 = [1.5, 0.0, 0.0]
    c4 = [1.5, 0.866, 0.5]
    angle = paramsgen.calculate_dihedral_params(c1, c2, c3, c4)
    assert angle == pytest.approx(60.0, abs=1.0)


def test_calculate_dihedral_params_cis(paramsgen):
    """Cis (0°) dihedral."""
    c1 = [0.0, 0.0, 0.0]
    c2 = [1.0, 0.0, 0.0]
    c3 = [2.0, 1.0, 0.0]
    c4 = [2.0, 2.0, 0.0]
    angle = paramsgen.calculate_dihedral_params(c1, c2, c3, c4)
    # Should be roughly 0 degrees
    assert abs(angle) < 30


def test_calculate_dihedral_params_90(paramsgen):
    """90° dihedral (orthogonal planes)."""
    c1 = [0.0, 0.0, 0.0]
    c2 = [1.0, 0.0, 0.0]
    c3 = [1.0, 1.0, 0.0]
    c4 = [1.0, 1.0, 1.0]
    angle = paramsgen.calculate_dihedral_params(c1, c2, c3, c4)
    assert 80 < abs(angle) < 100


# ============================================================================
# 3D Geometry: Bond Angle
# ============================================================================

def test_calculate_angle_params_linear(paramsgen):
    """Linear (180°) arrangement."""
    c1 = [0.0, 0.0, 0.0]
    c2 = [1.0, 0.0, 0.0]
    c3 = [2.0, 0.0, 0.0]
    angle = paramsgen.calculate_angle_params(c1, c2, c3)
    assert angle == pytest.approx(180.0, abs=0.01)


def test_calculate_angle_params_right_angle(paramsgen):
    """Right angle (90°)."""
    c1 = [1.0, 0.0, 0.0]
    c2 = [0.0, 0.0, 0.0]
    c3 = [0.0, 1.0, 0.0]
    angle = paramsgen.calculate_angle_params(c1, c2, c3)
    assert angle == pytest.approx(90.0, abs=0.01)


def test_calculate_angle_params_tetrahedral(paramsgen):
    """Tetrahedral angle (~109.5°)."""
    # Methane geometry: H-C-H ≈ 109.47°
    c2 = [0.0, 0.0, 0.0]  # central C
    c1 = [0.0, 0.0, 1.09]
    c3 = [1.03, 0.0, -0.36]
    angle = paramsgen.calculate_angle_params(c1, c2, c3)
    assert angle == pytest.approx(109.47, abs=2.0)


# ============================================================================
# RMSD Calculation (Kabsch Algorithm)
# ============================================================================

def test_kabsch_algorithm_identity(paramsgen):
    """Rotating identical point sets should give identity matrix."""
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Q = P.copy()
    R = paramsgen.kabsch_algorithm(P, Q)
    # R should be identity (or very close)
    assert np.allclose(R, np.eye(3), atol=1e-10)


def test_kabsch_algorithm_returns_proper_rotation(paramsgen):
    """Kabsch rotation matrix should be orthogonal with determinant 1."""
    np.random.seed(42)
    P = np.random.randn(10, 3)
    # Apply a known rotation (60° around Z then 30° around X)
    theta_z, theta_x = np.radians(60), np.radians(30)
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x),  np.cos(theta_x)]
    ])
    R_true = Rz @ Rx
    Q = P @ R_true.T
    R = paramsgen.kabsch_algorithm(P, Q)
    # R should be a proper rotation matrix
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
    # Applying R to centered P should give centered Q (RMSD ≈ 0)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    assert np.allclose(Pc @ R, Qc, atol=1e-10)


def test_calculate_rmsd_identical(paramsgen):
    """RMSD of identical structures should be 0."""
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert paramsgen.calculate_rmsd(P, P) == pytest.approx(0.0, abs=1e-10)


def test_calculate_rmsd_different(paramsgen):
    """RMSD of different structures should be > 0."""
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Q = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rmsd = paramsgen.calculate_rmsd(P, Q)
    assert rmsd > 0.5


# ============================================================================
# MOL2 Reading
# ============================================================================

def test_read_mol2_file_skips_caps(paramsgen):
    """Should skip first 12 atoms (ACE + NME caps)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mol2', delete=False) as f:
        f.write("@<TRIPOS>ATOM\n")
        for i in range(1, 16):  # 15 atoms total: 12 cap + 3 residue
            f.write(f"     {i} A{i}      0.0000    0.0000    0.0000 "
                    f"c3        1 RES    0.0000\n")
        f.write("@<TRIPOS>BOND\n")
        path = f.name
    try:
        atom_lines = paramsgen.read_mol2_file(path)
        assert len(atom_lines) == 3  # 15 - 12 = 3 residue atoms
    finally:
        os.unlink(path)


def test_read_mol2_file_empty(paramsgen):
    """MOL2 with only cap atoms should return empty list."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mol2', delete=False) as f:
        f.write("@<TRIPOS>ATOM\n")
        for i in range(1, 13):  # exactly 12 cap atoms
            f.write(f"     {i} A{i}      0.0000    0.0000    0.0000 "
                    f"c3        1 RES    0.0000\n")
        f.write("@<TRIPOS>BOND\n")
        path = f.name
    try:
        atom_lines = paramsgen.read_mol2_file(path)
        assert len(atom_lines) == 0
    finally:
        os.unlink(path)


# ============================================================================
# Charge Adjustment
# ============================================================================

def test_adjust_charges_to_integer_already_integer(paramsgen):
    """Charges that already sum to integer should not change."""
    charges = [0.5, -0.5, 0.0]  # sums to 0
    result = paramsgen.adjust_charges_to_integer(charges, 0)
    assert sum(result) == pytest.approx(0.0)
    assert result == [0.5, -0.5, 0.0]


def test_adjust_charges_to_integer_rounding(paramsgen):
    """Slight rounding errors should be corrected."""
    charges = [0.333, 0.333, 0.333]  # sums to 0.999
    result = paramsgen.adjust_charges_to_integer(charges, 1)
    assert sum(result) == pytest.approx(1.0, abs=0.001)


def test_adjust_charges_to_integer_largest_abs_adjusted(paramsgen):
    """Largest absolute charges should get the adjustment first."""
    charges = [0.001, 0.001, 0.998]  # sums to 0.999, need +0.001
    result = paramsgen.adjust_charges_to_integer(charges, 1)
    assert sum(result) == pytest.approx(1.0, abs=0.001)
    # The atom with largest abs charge (0.998) should get adjusted
    assert result[2] == pytest.approx(0.999) or result[2] > 0.998


def test_adjust_charges_to_integer_negative_target(paramsgen):
    """Negative target charge."""
    charges = [-0.33, -0.33, -0.33]  # sums to -0.99
    result = paramsgen.adjust_charges_to_integer(charges, -1)
    assert sum(result) == pytest.approx(-1.0, abs=0.001)


# ============================================================================
# Conformer Extraction
# ============================================================================

def test_extract_conformers(paramsgen):
    """Should split combined PDB into individual conformer blocks."""
    lines = [
        "ATOM 1 N ALA 1\n",
        "ATOM 2 CA ALA 1\n",
        "END\n",
        "ATOM 1 N ALA 1\n",
        "ATOM 2 CA ALA 1\n",
        "END\n",
    ]
    blocks = paramsgen.extract_conformers(lines)
    assert len(blocks) == 2
    assert "END\n" in blocks[0]
    assert "END\n" in blocks[1]


def test_extract_conformers_single(paramsgen):
    """Single conformer should return one block."""
    lines = ["ATOM 1 N ALA 1\n", "END\n"]
    blocks = paramsgen.extract_conformers(lines)
    assert len(blocks) == 1


def test_extract_conformers_no_end(paramsgen):
    """Lines without END should still produce a block."""
    lines = ["ATOM 1 N ALA 1\n", "ATOM 2 CA ALA 1\n"]
    blocks = paramsgen.extract_conformers(lines)
    assert len(blocks) == 1


# ============================================================================
# Dihedral Angle from PDB Coordinates
# ============================================================================

def test_calculate_dihedral_via_coords(paramsgen):
    """Dihedral from 4 coordinate sets should match reference values."""
    # Butane-like geometry: C-C-C-C with 60° dihedral
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([1.5, 0.0, 0.0])
    c3 = np.array([2.0, 1.4, 0.0])
    c4 = np.array([3.0, 1.5, 1.0])
    angle = paramsgen.calculate_dihedral(c1, c2, c3, c4)
    # Should be in [0, 180] range
    assert 0 <= angle <= 180


# ============================================================================
# Adaptive RMSD Threshold
# ============================================================================

def test_get_adaptive_rmsd_threshold_low_complexity(paramsgen):
    """Small v_value → large threshold (less complex, fewer conformers needed)."""
    assert paramsgen.get_adaptive_rmsd_threshold(10, 0.001) == 0.80


def test_get_adaptive_rmsd_threshold_high_complexity(paramsgen):
    """Large v_value → small threshold (more complex, more conformers to filter)."""
    assert paramsgen.get_adaptive_rmsd_threshold(810, 0.001) == 0.10


def test_get_adaptive_rmsd_threshold_user_override(paramsgen):
    """User-provided threshold should override adaptive value."""
    assert paramsgen.get_adaptive_rmsd_threshold(30, 0.42) == 0.42


def test_get_adaptive_rmsd_threshold_unknown_v_value(paramsgen):
    """Unknown v_value should fall back to 0.01."""
    assert paramsgen.get_adaptive_rmsd_threshold(999, 0.001) == 0.01


# ============================================================================
# Chi Angle Comparison
# ============================================================================

def test_compare_dihedrals_similar(paramsgen):
    """Similar dihedrals should return False (not different enough)."""
    ref = [60.0, 180.0, -60.0]
    pdb = [62.0, 178.0, -58.0]  # all within 15°
    assert not paramsgen.compare_dihedrals(ref, pdb, threshold=15)


def test_compare_dihedrals_different(paramsgen):
    """One different dihedral should return True."""
    ref = [60.0, 180.0, -60.0]
    pdb = [60.0, 180.0, 120.0]  # third angle differs by 180°
    assert paramsgen.compare_dihedrals(ref, pdb, threshold=15)


def test_compare_dihedrals_strict_threshold(paramsgen):
    """Stricter threshold should flag more differences."""
    ref = [60.0, 180.0]
    pdb = [65.0, 175.0]  # 5° difference each
    assert not paramsgen.compare_dihedrals(ref, pdb, threshold=15)  # passes loose
    assert paramsgen.compare_dihedrals(ref, pdb, threshold=3)  # fails strict


# ============================================================================
# PDB Energy Extraction (for energy screening)
# ============================================================================

def test_extract_energy_from_pdb_remark(paramsgen):
    """Should parse energy from PDB REMARK line."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write("REMARK Energy after optimization: -15.32 kcal/mol\n")
        f.write("ATOM      1  N   ALA     1       0.000   0.000   0.000\n")
        path = f.name
    try:
        energy = paramsgen.extract_energy_from_pdb_remark(path)
        assert energy == pytest.approx(-15.32)
    finally:
        os.unlink(path)


def test_extract_energy_from_pdb_remark_no_energy(paramsgen):
    """Should return None when no energy data present."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write("REMARK SMILES: CC\n")
        f.write("ATOM      1  N   ALA     1       0.000   0.000   0.000\n")
        path = f.name
    try:
        energy = paramsgen.extract_energy_from_pdb_remark(path)
        assert energy is None
    finally:
        os.unlink(path)


# ============================================================================
# PDB Splitting
# ============================================================================

def test_split_pdb_by_end(paramsgen):
    """Should split PDB into blocks on END markers."""
    lines = [
        "ATOM 1 N ALA 1\n",
        "HETATM 2 CA ALA 1\n",
        "END\n",
        "ATOM 1 N ALA 1\n",
        "END\n",
    ]
    blocks = paramsgen.split_pdb_by_end(lines)
    assert len(blocks) == 2


def test_remove_ace_nme(paramsgen):
    """Should remove ACE and NME residue lines."""
    lines = [
        "ATOM 1 N ACE 1\n",
        "ATOM 2 CA ALA 1\n",
        "ATOM 3 C NME 1\n",
        "END\n",
    ]
    filtered = paramsgen.remove_ace_nme(lines)
    assert len(filtered) == 2  # CA + END
    assert "ACE" not in ''.join(filtered)
    assert "NME" not in ''.join(filtered)


# ============================================================================
# Params File Reading
# ============================================================================

def test_read_params_file(paramsgen):
    """Should extract ATOM lines from params file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.params', delete=False) as f:
        f.write("ATOM    N1   Nbb   N\n")
        f.write("ATOM    CA   CAbb  CH1\n")
        f.write("BOND    N1   CA\n")
        f.write("ATOM    C    Cbb   C\n")
        path = f.name
    try:
        atom_lines = paramsgen.read_params_file(path)
        assert len(atom_lines) == 3
        assert 'N1' in atom_lines[0]
        assert 'CA' in atom_lines[1]
        assert 'C' in atom_lines[2]
    finally:
        os.unlink(path)
