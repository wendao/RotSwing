import sys
sys.path.append('/home/wendao/work/peprobe/aimnet/')
from aimnet import load_AIMNetMT_ens, load_AIMNetSMD_ens, AIMNetCalculator
from openbabel import pybel
import nglview
import ase
import ase.optimize
import ase.md
import numpy as np

def pybel2ase(mol):
    coord = np.asarray([a.coords for a in mol.atoms])
    numbers = np.asarray([a.atomicnum for a in mol.atoms])
    return ase.Atoms(positions=coord, numbers=numbers)

smiles = 'O=C([O-])[C@@H]([NH3+])Cc1c[nH]cn1'  # Histidine
mol = pybel.readstring("smi", smiles)
mol.make3D()
nglview.show_openbabel(mol.OBMol)

model_gas = load_AIMNetMT_ens().cuda()
model_smd = load_AIMNetSMD_ens().cuda()

calc_gas = AIMNetCalculator(model_gas)
calc_smd = AIMNetCalculator(model_smd)

atoms = pybel2ase(mol)

atoms.set_calculator(calc_gas)
opt = ase.optimize.BFGS(atoms, trajectory='gas_opt.traj')
opt.run(0.02)

traj = ase.io.Trajectory('gas_opt.traj', 'r')
nglview.show_asetraj(traj)

charges = calc_gas.results['elmoments'][0, :, 0]
print('Charges: ', charges)

volumes = calc_gas.results['volume'][0]
print('Volumes: ', volumes)

