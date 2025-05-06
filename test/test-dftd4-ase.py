from ase.build import molecule
from dftd4.ase import DFTD4
atoms = molecule('H2O')
atoms.calc = DFTD4(method="TPSS")
print(atoms.get_potential_energy())
