import os
import argparse
import re
import numpy as np
import shutil
from rdkit.Chem import Lipinski
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Superimposer
from rdkit.Chem import rdMolTransforms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
import random
import subprocess
from rdkit.Chem.rdMolTransforms import SetDihedralDeg
import time
import math
import sys
# 导入AIMNet库
'''
AIMNet库解析：
1. AIMNetCalculator
   一个能量和力的计算器，通过PyTorch导入预训练库进行处理
2. load_AIMNetMT_ens
   气相模型
3. load_AIMNetSMD_ens
   溶剂模型，此处采用的是一种通用溶剂，没有特定的溶剂类型
'''
sys.path.append('/home/wendao/work/peprobe/aimnet/')
from aimnet import load_AIMNetMT_ens, load_AIMNetSMD_ens, AIMNetCalculator
import ase
import ase.optimize
import ase.md
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from ase.constraints import FixInternals
import ase.io
import tempfile
from rdkit.Chem import SDMolSupplier, MolFromMolBlock
from rdkit.Chem import rdmolfiles

# 定义乙酰化和甲氨基化的SMARTS模式
acetylation_smarts = '[N:1][C:2][C:3](=[O:4])>>[C:5][C:7](=[O:6])[N:1][C:2][C:3](=[O:4])'    # 乙酰化，为主链N连上乙酰基[C:5][C:7](=[O:6])
amidation_smarts = '[C:1][C:2](=[O:3])[O:4]>>[C:1][C:2]([N:5][C:6])(=[O:3])'                 # 甲氨基化，为羧基主链C连上甲氨基([N:5][C:6])(=[O:3]。注意这里reactant里需要将单键O[O:4]标注出来，不然会报错

#根据smiles的电荷信息计算体系净电荷值
def calculate_system_charge(smiles_filepath):
    with open(smiles_filepath, 'r') as file:
        smiles = file.read().strip()
    
    # 统计SMILES中存在多少个正负号，然后简单做差即可计算体系电荷
    negative_count = smiles.count('-')
    positive_count = smiles.count('+')
    
    system_charge = positive_count - negative_count
    return system_charge
    
#为NCAA进行封端，并输出标准化的SMILES字符串
def process_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # 如果上一部转化SMILES读取失败，则报错误信息
    if mol is None:
        raise ValueError(f"Unable to parse SMILES: '{smiles}'")

    # 创建乙酰化和甲氨基化的反应
    rxn_acetylation = AllChem.ReactionFromSmarts(acetylation_smarts)
    rxn_amidation = AllChem.ReactionFromSmarts(amidation_smarts)

    # 应用甲氨基化反应
    products_amidation = rxn_amidation.RunReactants((mol,))
    if not products_amidation:
        raise ValueError("C-terminal amidation failed")

    product_amidation = products_amidation[0][0]

    # 应用乙酰化反应
    products_acetylation = rxn_acetylation.RunReactants((product_amidation,))
    if not products_acetylation:
        raise ValueError("N-terminal acetylation failed")

    product_acetylation = products_acetylation[0][0]

    # 获取乙酰化后的分子的 SMILES
    final_smiles = Chem.MolToSmiles(product_acetylation, canonical=False)
    return final_smiles

# 用于将RDKit对象转变为ASE的Atoms对象
def rdkit2ase(mol):
    # 使用RDkit生成三维模型
    AllChem.EmbedMolecule(mol)
    # 获取生成的分子模型的三维坐标
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    coords = np.array([conf.GetAtomPosition(i) for i in range(num_atoms)])
    # 获取每一个原子的符号
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    # 返回Atoms对象
    return Atoms(symbols=symbols, positions=coords)

# 用于计算指定二面角
def get_dihedral_angle(conf, indices):
    
    angle = rdMolTransforms.GetDihedralDeg(conf, *indices)
    return angle

# 用于进行二面角调整，主要功能是控制Phi，Psi为-150，150。该函数接受3个参数，Ⅰmol对象，Ⅱ组成二面角的四个原子索引，Ⅲ目标角度
def set_dihedral_angle(mol, indices, target_angle):

    # 从mol对象获取构象信息
    conf = mol.GetConformer()

    # 直接通过调整分子坐标来设置二面角
    rdMolTransforms.SetDihedralDeg(conf, *indices, target_angle)

# 检查函数，用于打印二面角信息，以检查二面角是否被正确设置
def print_dihedral_angle(mol, atom_indices):
    try:
        conf = mol.GetConformer()  # 获取分子的Conformer对象
        angle = get_dihedral_angle(conf, atom_indices)
        print(f"Dihedral angle for atoms {atom_indices}: {angle:.2f} degrees")
    except Exception as e:
        print(f"Error getting dihedral angle for atoms {atom_indices}: {e}")

# 用于将SMILES字符串通过obabel转换为RDKit分子对象
'''经过一系列尝试，发现Obabel的构象输出似乎比RDkit更稳定'''
def obabel_smiles_to_rdkit(smiles):
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp_sdf:
        # 使用obabel将SMILES字符串转换为3D结构，并保存为SDF文件
        obabel_cmd = f'obabel -:"{smiles}" --gen3D -O {tmp_sdf.name}'
        subprocess.run(obabel_cmd, shell=True, check=True)
        
        # 使用RDKit读取生成的SDF文件
        suppl = SDMolSupplier(tmp_sdf.name)
        mol = next(suppl)  # 获取第一个分子对象
    return mol

# 将rdkit对象保存为temp临时pdb文件
def save_rdkit_mol_to_pdb(mol, pdb_filename="temp.pdb"):

    # 创建保存PDB文件的文件夹
    pdb_dir = "pdb_files"
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)

    # 生成PDB文件的完整路径
    pdb_filepath = os.path.join(pdb_dir, pdb_filename)
    
    # 加氢
    mol = Chem.AddHs(mol)

    # 生成三维构象
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)
    
    # 将分子对象保存为PDB文件
    with open(pdb_filepath, 'w') as pdb_file:
        pdb_file.write(Chem.rdmolfiles.MolToPDBBlock(mol))
    
    print(f"RDKit mol 对象已保存为 {pdb_filename}")

# 用temp.pdb的原子名称替换AIMNet输出的pdb的原子名称，保证格式化输出
'''这一步的主要目的是衔接后续的原子重排，由于AIMNet有一套自己的编号规则，与RDkit不一致，因此结合此函数与上一步的save_rdkit_mol_to_pdb函数来对AIMNet的结果进行格式化修改，使其能够正常衔接后续操作'''
def replace_pdb_lines(name):
    # 定义文件路径
    smd_pdb_file = f"smd_{name}.pdb"
    temp_pdb_file = os.path.join("pdb_files", "temp.pdb")

    # 确保文件存在
    if not os.path.exists(smd_pdb_file):
        raise FileNotFoundError(f"File {smd_pdb_file} does not exist.")
    if not os.path.exists(temp_pdb_file):
        raise FileNotFoundError(f"File {temp_pdb_file} does not exist.")

    # 读取smd_{name}.pdb文件的内容到列表A
    with open(smd_pdb_file, 'r') as file:
        list_A = [line for line in file if line.startswith("ATOM") or line.startswith("HETATM")]

    # 读取pdb_files/temp.pdb文件的内容到列表B
    with open(temp_pdb_file, 'r') as file:
        list_B = [line for line in file if line.startswith("ATOM") or line.startswith("HETATM")]

    # 检查两个列表长度是否匹配
    if len(list_A) != len(list_B):
        raise ValueError("The number of 'ATOM' or 'HETATM' lines in the files do not match.")

    # 进行替换操作
    for i in range(len(list_B)):
        list_B[i] = list_B[i][:17] + list_A[i][17:]

    # 在列表B的末尾添加 "END" 行
    list_B.append("END\n")

    # 将修改后的列表B重新写入temp.pdb文件
    with open(temp_pdb_file, 'w') as file:
        file.writelines(list_B)

    print(f"File {temp_pdb_file} has been updated successfully.")

# AIMNet结构优化
'''调用AIMNet模块对NCAA进行结构优化，作为Gaussian结构优化的平替'''
def optimize_molecule(smiles, name):
    # 使用obabel生成3D构象，并将其转化为RDKit分子对象
    mol = obabel_smiles_to_rdkit(smiles)

    # 获取原子索引
    atom_indices = {atom.GetSymbol() + str(i+1): atom.GetIdx() for i, atom in enumerate(mol.GetAtoms())}
#    print(atom_indices)

    # 加载mol对象，使其转化为三维构象，同时将三维构象保存为temp.pdb文件
    AllChem.EmbedMolecule(mol)
    save_rdkit_mol_to_pdb(mol)
    
    # 优化前设置二面角为固定值
    '''这一步如果不做可能会导致后续AIMNet优化报错'''
    set_dihedral_angle(mol, [1, 3, 4, 5], -150.0)
    set_dihedral_angle(mol, [3, 4, 5, 7], 150.0)
    set_dihedral_angle(mol, [2, 1, 3, 4], 4.18)
    set_dihedral_angle(mol, [6, 5, 7, 8], 1.17)

    # 使用UFF力场进行结构优化
    '''初步优化分子结构，防止其因为结构过于糟糕导致AIMNet优化报错'''
    ff = UFFGetMoleculeForceField(mol)
    ff.Initialize()
    ff.Minimize()

    # 优化后重设二面角
    '''重新将主链上各二面角设置为定值，防止后续AIMNet自动调整二面角时报错'''
    set_dihedral_angle(mol, [1, 3, 4, 5], -150.0)
    set_dihedral_angle(mol, [3, 4, 5, 7], 150.0)
    set_dihedral_angle(mol, [2, 1, 3, 4], 4.18)
    set_dihedral_angle(mol, [6, 5, 7, 8], 1.17)
    
    # 检查二面角是否被成功设置
    print_dihedral_angle(mol, [1, 3, 4, 5])
    print_dihedral_angle(mol, [3, 4, 5, 7])
    print_dihedral_angle(mol, [2, 1, 3, 4])
    print_dihedral_angle(mol, [6, 5, 7, 8])    

    # 添加氢原子
    mol = Chem.AddHs(mol)
    # 将调整后的分子对象转化为ASE的Atoms对象
    atoms = rdkit2ase(mol)
    
    # 定义需要被固定的二面角
    dihedrals_deg = [
        [-150.0, [atom_indices['C2'], atom_indices['N4'], atom_indices['C5'], atom_indices['C6']]],
        [150.0, [atom_indices['N4'], atom_indices['C5'], atom_indices['C6'], atom_indices['N8']]]
    ]

    # 对上述二面角添加限制措施，并将限制信息传递给atoms对象
    constraints = FixInternals(dihedrals_deg=dihedrals_deg, epsilon=0.03)
    atoms.set_constraint(constraints)

    # 溶剂下优化
    atoms = rdkit2ase(mol)
    
    # 设置气相与溶剂模型，并提交至CUDA
    model_gas = load_AIMNetMT_ens().cuda()
    model_smd = load_AIMNetSMD_ens().cuda()
    calc_gas = AIMNetCalculator(model_gas)
    calc_smd = AIMNetCalculator(model_smd)
    
    # 气象优化
    atoms.calc = calc_gas
    opt = ase.optimize.BFGS(atoms, trajectory='gas_opt.traj')
    #opt.run(steps=499, fmax=0.001)
    opt.run(steps=1, fmax=0.001)
    
    # 从气相结果中提取电荷分布信息以及体积信息
    charges = calc_gas.results['elmoments'][0, :, 0]
    print('Charges: ', charges)
    volumes = calc_gas.results['volume'][0]
    print('Volumes: ', volumes)
    
    # 将电荷与原子名称写入 charge.txt 文件
    '''该charge文件在后续并没有被实际应用，最初的想法是用这里产生的charge电荷直接替换params的初始电荷，以省去RESP拟合步骤，但最终发现该电荷数值存在较大问题，遂放弃。'''
    with open('charge.txt', 'w') as charge_file:
        for atom, charge in zip(mol.GetAtoms(), charges):
            atom_name = atom.GetSymbol()  # 获取原子名称
            atom_index = atom.GetIdx() + 1  # 获取原子索引（1-based）
            charge_file.write(f"{atom_name}{atom_index}: {charge:.6f}\n")

    # 溶剂下优化
    atoms = rdkit2ase(mol)
    
    atoms.calc = calc_smd
    opt = ase.optimize.BFGS(atoms, trajectory='smd_opt.traj')
    opt.run(steps=499, fmax=0.001)
    
    # 将优化结果转换为PDB和SDF文件
    traj_files = {
        'gas': 'gas_opt.traj',
        'smd': 'smd_opt.traj'
    }
    
    for key, traj_file in traj_files.items():
        pdb_file = f"{key}_{name}.pdb"
        traj = ase.io.read(traj_file, index=-1)  # 只读取最后一帧
        ase.io.write(pdb_file, traj)
        final_pdb = os.path.join("pdb_files", "temp.pdb")

        # 读取生成的PDB文件并使用RDKit的原子命名规则进行重新命名
        with open(pdb_file, 'r') as f:
            pdb_lines = f.readlines()
    
        with open(pdb_file, 'w') as f:
            # 重原子和氢原子计数，用于将指定原子进行重新排列
            atom_count = 0
            hydrogen_count = 0
            for line in pdb_lines:
                if line.startswith("HETATM") or line.startswith("ATOM"):
                    atom_idx = int(line[6:11].strip()) - 1  # 获取原子的索引 (0-based)
                    atom = mol.GetAtomWithIdx(atom_idx)
                    atom_name = f"{atom.GetSymbol()}{atom.GetIdx()+1}".ljust(4)  # 使用RDKit的命名规则
                    line = line[:12] + atom_name + line[16:]  # 替换原子名称

                    atom_count += 1
                    is_hydrogen = line[76:78].strip() == 'H'  # 判断是否为氢原子
                    if is_hydrogen:
                        hydrogen_count += 1
                    
                    if atom_count <= 3:
                        # 前三行原子的氨基酸名命名为ACE
                        line = line[:17] + "ACE" + line[20:]
                    elif atom_count == 8 or atom_count == 9:
                        # 第八和第九行原子的氨基酸名命名为NME
                        line = line[:17] + "NME" + line[20:]
                    elif is_hydrogen:
                        if hydrogen_count <= 3:
                            # 前三行H原子的氨基酸名命名为ACE
                            line = line[:17] + "ACE" + line[20:]
                        elif 6 <= hydrogen_count <= 9:
                            # 第六到第九行H原子的氨基酸名命名为NME
                            line = line[:17] + "NME" + line[20:]
                        else:
                            # 其他所有没有调整的氢原子的氨基酸名命名为输入的字符串
                            line = line[:17] + name + line[20:]
                    else:
                        # 其他所有没有调整的原子的氨基酸名命名为输入的字符串
                        line = line[:17] + name + line[20:]
                f.write(line)

    replace_pdb_lines(name)

    return final_pdb
    
# 将SMILES字符串转换为PDB文件，并传递至generated_files对象
def smiles_to_pdb(smiles_list, names_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(names_list) < len(smiles_list):
        names_list.extend(['UAA'] * (len(smiles_list) - len(names_list)))
    elif len(names_list) > len(smiles_list):
        raise ValueError("The number of provided names exceeds the number of SMILES strings.")

    generated_files = []
    for i, (smiles, name) in enumerate(zip(smiles_list, names_list)):
        final_pdb = optimize_molecule(smiles, name)
      
    generated_files.append(final_pdb)

    return generated_files

# 对RDKit生成的构象进行原子重排，格式化输出以对接后续操作
def process_pdb_file(filepath, n_value, output_filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # 过滤掉不是以 "HETATM" 或 "ATOM  " 开头的行
    lines = [line for line in lines if line.startswith("HETATM") or line.startswith("ATOM  ")]

    # 字典存储需要进行调整的行
    line_dict = { "C2": None, "O1": None, "C1": None, "H1": None, "H2": None, "H3": None,
                  "N2": None, "C5": None, "H6": None, "H7": None, "H8": None, "H9": None,
                  "N1": None, "C3": None, "C4": None, "O2": None, "H4": None, "H5": None }
    other_lines = []

    for line in lines:
        atom_name = line[12:16].strip()
        if atom_name in line_dict:
            line_dict[atom_name] = line
        else:
            other_lines.append(line)

    # 按指定顺序重新排列行
    reordered_lines = []
    for key in ["C2", "O1", "C1", "H1", "H2", "H3", "N2", "C5", "H6", "H7", "H8", "H9", "N1", "C3", "C4", "O2"]:
        if line_dict[key] is not None:
            reordered_lines.append(line_dict[key])

    reordered_lines.extend(other_lines)

    #后置主链H原子
    if line_dict["H4"] is not None:
        reordered_lines.append(line_dict["H4"])

    if line_dict["H5"] is not None:
        reordered_lines.append(line_dict["H5"])

    # 修改氨基酸名称，指认出ACE和NME的部分
    for i in range(len(reordered_lines)):
        if i < 6:
            reordered_lines[i] = reordered_lines[i][:17] + "ACE" + reordered_lines[i][20:]
        elif i < 12:
            reordered_lines[i] = reordered_lines[i][:17] + "NME" + reordered_lines[i][20:]
        else:
            reordered_lines[i] = reordered_lines[i][:17] + n_value + reordered_lines[i][20:]

    # 添加 "END" 行
    reordered_lines.append("END\n")

    # 写回新文件
    with open(output_filepath, 'w') as file:
        file.writelines(reordered_lines)

# 遍历所有pdb，对其执行重排操作
def process_files(filepaths, n_values, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, (filepath, n_value) in enumerate(zip(filepaths, n_values)):
        output_filepath = os.path.join(output_dir, f'{n_value}_{i+1}.pdb')
        process_pdb_file(filepath, n_value, output_filepath)
        print(f"Processed {filepath}")

        # 读取并打印PDB文件的内容
        with open(output_filepath, 'r') as output_print:
            output_content = output_print.read()
            print(output_content)

# PDB_to_mol，并对mol文件进行注释
def process_pdb_files(input_file, pdb_dir):

    # 读取SMILES与PDB文件
    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
    
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]

    # 创建空字典，用于记录手性信息
    '''这部分功能尚未开发，原先的想法是对L型，D型，Peptoid甚至β，γ氨基酸进行识别，然后分而治之。由于目前只支持L型氨基酸，因此输出结果仅有L型。其余功能有待后续开发'''
    chirality_dict = {}

    for pdb_file in pdb_files:
        base_name = os.path.splitext(pdb_file)[0]
        idx = int(base_name.split('_')[-1]) - 1  
        smiles = smiles_list[idx]

        # 判断氨基酸为L型还是D型
        '''识别方式需要调整，最好通过对封端后的SMILES进行解析来实现这个功能，而不是根据手性碳来识别'''
        if '[C@@H]' in smiles:
            chirality_dict[pdb_file] = 'L'
        else:
            chirality_dict[pdb_file] = 'L'
        
        mol_output_dir = 'mol'
        if not os.path.exists(mol_output_dir):
            os.makedirs(mol_output_dir)
        
        # 将PDB文件转化为MOL文件
        pdb_filepath = os.path.join(pdb_dir, pdb_file)
        mol_filepath = os.path.join(mol_output_dir, f'{base_name}_opt.mol')
        command = f'obabel -i pdb {pdb_filepath} -o mol -O {mol_filepath}'
        os.system(command)
        
        with open(mol_filepath, 'r') as mol_file:
            mol_lines = mol_file.readlines()
        
        # 删除最后一行 "M  END"
        if mol_lines[-1].strip() == "M  END":
            mol_lines = mol_lines[:-1]

        # 根据氨基酸类型添加原子指认信息
        if chirality_dict[pdb_file] == 'L':
            mol_lines.extend([
                "M  ROOT 13\n",
                "M  POLY_N_BB 13\n",
                "M  POLY_CA_BB 14\n",
                "M  POLY_C_BB 15\n",
                "M  POLY_O_BB 16\n",
                "M  POLY_IGNORE 2 3 4 5 6 8 9 10 11 12\n",
                "M  POLY_UPPER 7\n",
                "M  POLY_LOWER 1\n",
                "M  POLY_PROPERTIES PROTEIN L_AA ALPHA_AA\n",
                "M  END\n"
            ])
        
        with open(mol_filepath, 'w') as mol_file:
            mol_file.writelines(mol_lines)

        print(f"Processed and saved {mol_filepath}")

        # 读取并打印mol文件的内容
        with open(mol_filepath, 'r') as mol_print:
            mol_content = mol_print.read()
            print(mol_content)
            
# 调用molfile_to_params_polymer.py脚本对mol文件进行参数化，生成params参数文件
def molfile_to_params(mol_filepath, name):
    # 创建params文件的完整路径
    base_name = os.path.splitext(os.path.basename(mol_filepath))[0]
    params_filename = f"{base_name}.params"
    params_filepath = os.path.join(os.getcwd(), params_filename)

    # 检查参数文件是否已经存在，如果存在则skip
    if os.path.exists(params_filepath):
        print(f"Params file {params_filepath} already exists. Skipping conversion.")
        return
    
    # 通过命令行调用molfile_to_params_polymer.py脚本，执行参数化
    # 实际使用中需要根据当前操作环境下的脚本路径对以下命令进行修改
    script_dir = os.path.dirname(os.path.abspath(__file__))
    command = f"python2 {script_dir}/molfile_to_params_polymer_modify.py -n {name} --polymer {mol_filepath}"
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # 检查任务是否正确运行
    if result.returncode != 0:
        print(f"Error converting {mol_filepath} to params file. Command output:\n{result.stdout}\n{result.stderr}")
        return

    # 打印成功信息
    print(f"Converted {mol_filepath} to {params_filepath}")

# 创建temp文件，用于后续的RESP电荷替换
def molfile_to_params_temps(mol_filepath, name):
    # 创建temps.params路径
    base_name = os.path.splitext(os.path.basename(mol_filepath))[0]
    params_filename = f"{base_name}_temps.params"
    params_filepath = os.path.join(os.getcwd(), params_filename)

    # 检查参数文件是否已经存在
    if os.path.exists(params_filepath):
        print(f"Params file {params_filepath} already exists. Skipping conversion.")
        return
    
    # 使用molfile_to_params_polymer_modify.py脚本进行参数化
    script_dir = os.path.dirname(os.path.abspath(__file__))
    command = f"python2 {script_dir}/molfile_to_params_polymer_modify.py -n {name}_temps --no_reorder --polymer {mol_filepath}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # 检查任务是否被顺利运行
    if result.returncode != 0:
        print(f"Error converting {mol_filepath} to params file. Command output:\n{result.stdout}\n{result.stderr}")
        return

    # 打印运行成功信息
    print(f"Converted {mol_filepath} to {params_filepath}")

# GJF修改函数，为其添加内存，核数以及二面角固定信息 
def modify_gjf_file(resp_folder, system_charge):
    # 遍历目录中的所有文件
    for filename in os.listdir(resp_folder):
        if filename.endswith('.gjf'):
            gjf_filepath = os.path.join(resp_folder, filename)
            
            # 读取 .gjf 文件
            with open(gjf_filepath, 'r') as file:
                lines = file.readlines()

            # 检索并修改电荷行，为 .gjf 文件提供正确的电荷信息
            for i in range(len(lines)):
                if lines[i].strip() == "0   1":
                    lines[i] = f"{system_charge}   1\n"
                    break

            # 删除第一、第二行，并定义内存和核数。可根据需要自行更改
            if len(lines) > 1:
                lines[0] = "%mem=100GB\n"
                lines[1] = "%nprocshared=24\n"
            else:
                lines.insert(0, "%mem=100GB\n")
                lines.insert(1, "%nprocshared=24\n")

            # 在最后一行非空行后添加固定二面角信息，以确保结构优化时Phi，Psi角固定不动
            non_empty_lines = [line for line in lines if line.strip()]
            last_non_empty_line_index = lines.index(non_empty_lines[-1])
            lines.insert(last_non_empty_line_index + 1, '\n13 14 15 7 F\n1 13 14 15 F\n\n\n\n')

            # 将修改后的内容写回到 .gjf 文件
            with open(gjf_filepath, 'w') as file:
                file.writelines(lines)

            print(f"Modified {gjf_filepath}")

# 提交RESP电荷拟合任务，并将计算结果转化为mol2文件
def generate_opt(pdb_file, resp,resp_folder, system_charge):
    # 使用 antechamber 将 PDB 文件转换为 Gaussian RESP 计算的 gjf 文件
    os.system(f'antechamber -i {pdb_file} -fi pdb -o {resp}.gjf -fo gcrt -gk "# HF/6-31G*  SCF=Tight  Pop=MK  iop(6/33=2,  6/41=10, 6/42=15)"')
    print(f'\nGaussian RESP-calculation input file ({resp}.gjf) for {pdb_file} has already been generated by antechamber!!\n')
    
    # 编辑 gjf 文件
    modify_gjf_file(resp_folder, system_charge)
    
    # 运行 Gaussian 计算并将输出转换为 mol2 文件
    os.system(f'/home/wendao/install/g16/g16 {resp}.gjf && antechamber -i {resp}.log -fi gout -o {resp}.mol2 -fo mol2 -at amber -pf y -c resp')
    print(f'\nThe optimized structure with RESP charge has been output to {resp}.mol2 and needs to be further processed!!\n')

# 使用RESP电荷替换params文件的原始电荷
def params_charge_adjust(resp_folder):
    # 遍历RESP文件夹中的mol2文件
    for mol2_file in os.listdir(resp_folder):
        if mol2_file.endswith('.mol2'):
            # 提取文件名前三个字符作为res对象
            res = mol2_file[:3]

            # 查找对应的params文件
            params_file = f'{res}_temps.params'
            if not os.path.exists(params_file):
                print(f"Warning: {params_file} not found for {mol2_file}")
                continue
            
            # 读取params文件中ATOM开头行，将其全部保存进一个列表
            params_atom_lines = []
            with open(params_file, 'r') as f_params:
                for line in f_params:
                    if line.startswith('ATOM'):
                        params_atom_lines.append(line)
            
            # 读取mol2文件内容
            with open(os.path.join(resp_folder, mol2_file), 'r') as f_mol2:
                mol2_lines = f_mol2.readlines()
            
            start_idx = -1
            end_idx = -1
            
            # 找到@<TRIPOS>ATOM和@<TRIPOS>BOND之间的行的索引范围
            try:
                start_idx = mol2_lines.index('@<TRIPOS>ATOM\n') + 1
                end_idx = mol2_lines.index('@<TRIPOS>BOND\n')
            except ValueError:
                print(f"Error: Unable to find '@<TRIPOS>ATOM' or '@<TRIPOS>BOND' in {mol2_file}")
                continue
    
            # 检索这些行中第一列>=13的行，并将它们保存进另一个列表
            mol2_atom_lines_to_replace = []
            for i in range(start_idx, end_idx):
                parts = mol2_lines[i].split()
                if len(parts) >= 1:
                    try:
                        atom_index = int(parts[0])
                    except ValueError:
                        continue  # 如果无法转换为整数，跳过该行
            
                    if atom_index >= 13:
                        mol2_atom_lines_to_replace.append((i, mol2_lines[i]))
    
            # 将params_atom_lines中的[5:9]部分覆盖mol2_atom_lines_to_replace的[7:11]部分。即使用params的原子名称来替代mol2文件中的原子名称
            for j, (i, line) in enumerate(mol2_atom_lines_to_replace):
                parts = list(line)
                if len(parts) >= 11 and j < len(params_atom_lines):
                    new_value = params_atom_lines[j][5:9]
                    parts[7:11] = new_value
                    mol2_lines[i] = ''.join(parts)
            
            # 写入更新后的mol2文件
            output_file_path = os.path.join(resp_folder, mol2_file)
            with open(output_file_path, 'w') as f_mol2:
                f_mol2.writelines(mol2_lines)
                
                    # 读取更新后的mol2文件内容
            with open(output_file_path, 'r') as f_updated_mol2:
                updated_mol2_lines = f_updated_mol2.readlines()
            
            # 查找@<TRIPOS>ATOM和@<TRIPOS>BOND之间的行的索引范围
            try:
                start_idx = updated_mol2_lines.index('@<TRIPOS>ATOM\n') + 1
                end_idx = updated_mol2_lines.index('@<TRIPOS>BOND\n')
            except ValueError:
                print(f"Error: Unable to find '@<TRIPOS>ATOM' or '@<TRIPOS>BOND' in updated {mol2_file}")
                continue
            
            # 检查并调整列表中[7:8]为数字的元素，将数字移至字母后面，使原子名称符合gromacs字母在前数字在后的规范
            adjusted_lines = []
            for i in range(start_idx, end_idx):
                line = updated_mol2_lines[i]
                parts = list(line)
                if len(parts) >= 8 and parts[7].isdigit():
                    if len(parts) >= 11 and parts[9] != ' ' and parts[10] != ' ':
                        parts[11] = parts[7]
    #                    print(parts[11])
                    if len(parts) >= 10 and parts[9] != ' ':
                        parts[10] = parts[7]
    #                    print(parts[10])
                    if len(parts) >= 9 and parts[9] == ' ':
                        parts[9] = parts[7]
    #                    print(parts[9])
                    parts[7:8] = ' '
                    adjusted_lines.append(''.join(parts))
                else:
                    adjusted_lines.append(line)

            for i in range(start_idx, end_idx):
                mol2_lines[i] = adjusted_lines[i - start_idx]
                
            # 写入更新后的mol2文件
            with open(output_file_path, 'w') as f_final_mol2:
                f_final_mol2.writelines(mol2_lines)
            
            print(f"Processed {mol2_file} successfully.")

            # 读取并打印mol2文件的内容
            with open('RESP/'+mol2_file, 'r') as mol2_print:
                mol2_content = mol2_print.read()
                print(mol2_content)

# 计算封端电荷
def calculate_capcharge(resp_folder):

    for mol2_file in os.listdir(resp_folder):
        if mol2_file.endswith('.mol2'):
            # 提取文件名前三个字符作为res对象
            res = mol2_file[:3]
            
            # 读取mol2文件内容
            with open(os.path.join(resp_folder, mol2_file), 'r+') as f_mol2:
                mol2 = f_mol2.readlines()
                
                # 定位ATOM所在行
                end = mol2.index('@<TRIPOS>BOND\n') 
                start = mol2.index('@<TRIPOS>ATOM\n')

                ace_cap, nme_cap = 0, 0

                # 计算ACE封端电荷
                for i in range(1 + start, 7 + start):
                    lis = list(filter(None, mol2[i].replace('\n', '').split(' ')))
                    ace_cap += eval(lis[-1])
                
                # 计算NME封端电荷
                for j in range(7 + start, 13 + start):
                    lis = list(filter(None, mol2[j].replace('\n', '').split(' ')))
                    nme_cap += eval(lis[-1])

                # 四舍五入电荷值，以保留六位小数
                ace_cap, nme_cap = round(ace_cap, 6), round(nme_cap, 6)
                
                # 写入封端电荷信息
                charge_info = open(f'{res}_cap.charge', 'w')
                charge_info.write(f'ace_cap: {ace_cap}\n')
                print(f'Sum charge of ACE: {ace_cap}')
                charge_info.write(f'nme_cap: {nme_cap}\n\n')
                print(f'Sum charge of NME: {nme_cap}')
                charge_info.close()
                
                # 删除生成的电荷信息文件
                os.remove(f'{res}_cap.charge')
                
                # N端与ACE封端电荷相加
                N_ncaa_line = mol2[start + 13]
                N_ncaa = list(filter(None, N_ncaa_line.replace('\n', '').split(' ')))
                if N_ncaa[1] == 'N':
                    print(f'Detect N-termini, with original charge {N_ncaa[-1]}')
                    new_N_charge = round(eval(N_ncaa_line[-10:-1]) + ace_cap, 6)
                    new_N_charge_str = f'{new_N_charge:9.6f}'  # 确保电荷值的格式化
                    mol2[start + 13] = N_ncaa_line.replace(N_ncaa_line[-10:-1], str(new_N_charge), 1)
                    print(f'Update N-termini charge with new value {new_N_charge}')
                else:
                    raise Exception("Please check your PDB input, ensure N-CA-C-O order.")

                # C端与NME封端电荷相加
                C_ncaa_line = mol2[start + 15]
                C_ncaa = list(filter(None, C_ncaa_line.replace('\n', '').split(' ')))
                if C_ncaa[1] == 'C':
                    print(f'Detect C-termini, with original charge {C_ncaa[-1]}')
                    new_C_charge = round(eval(C_ncaa_line[-10:-1]) + nme_cap, 6)
                    new_C_charge_str = f'{new_C_charge:9.6f}'  # 确保电荷值的格式化
                    mol2[start + 15] = C_ncaa_line.replace(C_ncaa_line[-9:-1], str(new_C_charge), 1)
                    print(f'Update C-termini charge with new value {new_C_charge}')
                else:
                    raise Exception("Please check your PDB input, ensure N-CA-C-O order.")

# 计算params二面角，用于排除params原子树构建时可能出现的bug
def calculate_dihedral_params(coords1, coords2, coords3, coords4):
    def vector_subtract(a, b):
        return [a[i] - b[i] for i in range(3)]

    def vector_dot(a, b):
        return sum(a[i] * b[i] for i in range(3))

    def vector_cross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ]

    def vector_magnitude(v):
        return math.sqrt(sum(x**2 for x in v))

    def vector_normalize(v):
        mag = vector_magnitude(v)
        return [x / mag for x in v]

    p = coords1
    q = coords2
    r = coords3
    s = coords4

    pq = vector_subtract(q, p)
    qr = vector_subtract(r, q)
    rs = vector_subtract(s, r)

    pq_cross_qr = vector_cross(pq, qr)
    qr_cross_rs = vector_cross(qr, rs)

    pq_cross_qr_mag = vector_magnitude(pq_cross_qr)
    qr_cross_rs_mag = vector_magnitude(qr_cross_rs)

    pq_cross_qr_dot_qr_cross_rs = vector_dot(pq_cross_qr, qr_cross_rs)
    cos_dihedral = pq_cross_qr_dot_qr_cross_rs / (pq_cross_qr_mag * qr_cross_rs_mag)

    qr_normalized = vector_normalize(qr)
    pq_cross_qr_cross_qr = vector_cross(pq_cross_qr, qr_normalized)
    sin_dihedral = vector_dot(pq_cross_qr_cross_qr, qr_cross_rs) / (pq_cross_qr_mag * qr_cross_rs_mag)

    dihedral = math.degrees(math.atan2(sin_dihedral, cos_dihedral))
    return dihedral

# 计算params角，用于排除params原子树构建时可能出现的bug
def calculate_angle_params(coords1, coords2, coords3):
    a = coords1
    b = coords2
    c = coords3

    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]

    ba_dot_bc = sum([ba[i] * bc[i] for i in range(3)])
    ba_mag = math.sqrt(sum([ba[i]**2 for i in range(3)]))
    bc_mag = math.sqrt(sum([bc[i]**2 for i in range(3)]))

    cos_angle = ba_dot_bc / (ba_mag * bc_mag)
    return math.degrees(math.acos(cos_angle))

# 检查params文件，如果发现角/二面角异常则修复他们
def process_params_file(n_value):
    filename = f'{n_value}.params'
    
    # 进入 RESP 文件夹并转换 mol2 文件为 pdb 文件
    mol2_file = f'RESP/{n_value}_1.mol2'
    pdb_file = f'RESP/{n_value}_1.pdb'
    os.system(f'obabel {mol2_file} -O {pdb_file}')
    
    # 读取生成的 PDB 文件并提取原子坐标
    atom_coords = {}
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                parts = line.split()
                atom_name = parts[2]
                coords = [float(parts[6]), float(parts[7]), float(parts[8])]
                atom_coords[atom_name] = coords

    # 计算所需值
    value_1 = calculate_dihedral_params(atom_coords['O'], atom_coords['C'], atom_coords['CA'], atom_coords['N1'])
    value_2 = 180 - calculate_angle_params(atom_coords['O'], atom_coords['C'], atom_coords['CA'])
    value_3 = math.sqrt(sum([(atom_coords['O'][i] - atom_coords['C'][i])**2 for i in range(3)]))
    
    value_4 = calculate_dihedral_params(atom_coords['CB'], atom_coords['CA'], atom_coords['N'], atom_coords['C'])
    value_5 = 180 - calculate_angle_params(atom_coords['CB'], atom_coords['CA'], atom_coords['N'])
    value_6 = math.sqrt(sum([(atom_coords['CB'][i] - atom_coords['CA'][i])**2 for i in range(3)]))
    
    # 读取params文件内容
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # 查找并处理以“ICOOR_INTERNAL    O ”开头的行
    for i, line in enumerate(lines):
        if line.startswith('ICOOR_INTERNAL    O '):
            parts = line.split()
            if len(parts) == 8:
                dihedral_angle = float(parts[2])
                bond_angle = float(parts[3])
                
                if not (179 <= dihedral_angle <= 181 or -181 <= dihedral_angle <= -179) or not (59 <= bond_angle <= 61):
                    new_line = f'ICOOR_INTERNAL    O    {value_1:.6f}   {value_2:.6f}    {value_3:.6f}   C     CA  UPPER\n'
                    lines[i] = new_line
                    break
                    
    # 查找并处理以“ICOOR_INTERNAL    CB”开头的行
    for i, line in enumerate(lines):
        if line.startswith('ICOOR_INTERNAL    CB'):
            parts = line.split()
            if len(parts) == 8:
                dihedral_angle = float(parts[2])
                bond_angle = float(parts[3])
                
                if not (-123 <= dihedral_angle <= -121 or 121 <= dihedral_angle <= 123) or not (69 <= bond_angle <= 71):
                    new_line = f'ICOOR_INTERNAL    CB  -{value_4:.6f}   {value_5:.6f}    {value_6:.6f}   CA    N     C \n'
                    lines[i] = new_line
                    break
    
    # 将修改后的内容写回文件
    with open(filename, 'w') as file:
        file.writelines(lines)
    
    # 打印params文件的内容
    with open(filename, 'r') as params_print:
        params_content = params_print.read()
        print(params_content)

# 读取mol2文件的原子所在行
def read_mol2_file(mol2_filepath):
    atom_lines = []
    with open(mol2_filepath, 'r') as f:
        lines = f.readlines()
    
    atom_started = False
    for line in lines:
        if line.startswith('@<TRIPOS>ATOM'):
            atom_started = True
            continue
        if line.startswith('@<TRIPOS>BOND'):
            atom_started = False
            continue
        if atom_started and line.strip():  
            atom_lines.append(line.strip())
    
    return atom_lines[12:]  

# 读取params的原子所在行
def read_params_file(params_filepath):
    atom_lines = []
    with open(params_filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('ATOM'):
            atom_lines.append(line.strip())
    
    return atom_lines

# 电荷置零函数，保证params的电荷总和为整数
def adjust_charges_to_integer(charges_list,system_charge):
    #计算当前净电荷距离整数电荷的差值
    rounded_charges = [round(charge, 2) for charge in charges_list]

    charge_diff = system_charge - sum(rounded_charges)
    
    if charge_diff == 0:
        return rounded_charges

    # 按照电荷绝对值从大到小排序
    sorted_indices = sorted(range(len(charges_list)), key=lambda i: abs(charges_list[i]), reverse=True)

    # 按电荷从大到小依次进行±0.01的调整，从而在对体系电荷产生最小影响的情况下使电荷值为整数
    for _ in range(abs(int(charge_diff * 100))):  # 需要的调整次数
        for i in sorted_indices:
            if charge_diff > 0:
                rounded_charges[i] += 0.01
                charge_diff -= 0.01
            elif charge_diff < 0:
                rounded_charges[i] -= 0.01
                charge_diff += 0.01
            if round(charge_diff, 2) == 0:
                break
    
    return rounded_charges

# 修正params原子电荷，使其为整数
def modify_params_file(params_filepath, rounded_charges):
    with open(params_filepath, 'r') as f:
        lines = f.readlines()

    atom_lines = []
    atom_indices = []

    #读取params文件的ATOM行
    for idx, line in enumerate(lines):
        if line.startswith('ATOM'):
            atom_lines.append(line.strip())
            atom_indices.append(idx)

    #检查原子数和电荷列表数是否匹配
    if len(atom_lines) != len(rounded_charges):
        print(f"ATOM lines count: {len(atom_lines)}")
        print(f"Rounded charges count: {len(rounded_charges)}")
        raise ValueError('params 文件中的 ATOM 行数与电荷列表长度不匹配。')

    for i in range(len(atom_lines)):
        original_line = atom_lines[i]
        # 提取新的电荷值
        charge_part = original_line[-5:]  # 提取最后的电荷部分
        new_charge = f"{rounded_charges[i]:.2f}"
        
        # 检查正数电荷值并在前面加空格以保证格式统一
        if float(new_charge) > 0:
            new_charge = f" {new_charge}"
        
        new_line = original_line[:-5] + new_charge + '\n'  # 应用新的电荷值组装新的行内容
        lines[atom_indices[i]] = new_line  # 更新原始文件中的ATOM行

    # 将修改后的 ATOM 行写回文件
    with open(params_filepath, 'w') as f:
        f.writelines(lines)

# 将修改后的电荷信息更新到params文件
'''因原子匹配问题，上述操作均在temps文件中进行，此函数的功能是将修改好的电荷信息传递给params文件'''
def update_params_file_with_temps(temps_filepath, params_filepath):
    # 读取params文件中的所有行
    with open(temps_filepath, 'r') as f:
        params_lines = f.readlines()

    # 读取temps文件中的所有以ATOM开头的行
    temps_atoms = []
    with open(params_filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                temps_atoms.append(line.strip())

    # 遍历params文件中的所有以ATOM开头的行，进行替换
    updated_params_lines = []
    for line in params_lines:
        if line.startswith('ATOM'):
            atom_id = line[:18]
            for temps_line in temps_atoms:
                if temps_line[:18] == atom_id:
                    line = temps_line + '\n'
                    break
        updated_params_lines.append(line)

    # 将更新后的行写回params文件
    with open(temps_filepath, 'w') as f:
        f.writelines(updated_params_lines)

    # 读取并打印params文件的内容
    with open(temps_filepath, 'r') as params_new_print:
        params_new_content = params_new_print.read()
        print(params_new_content)

# mol2文件检查函数
'''该函数的功能是检查mol2文件中的原子名称，有些名称gromacs无法识别，导致MD参数化失败。目前发现绝大对数参数化报错均是由于原子无法正确识别导致的，因此需要将其改为MD认识的原子名称。目前仅添加了叠氮的异常处理，如果遇到其他原子识别异常，可以将异常处理信息添加在该函数中'''
def process_mol2_file(file_path):
    # 读取mol2文件
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 将部分显示为DU或N3的N原子的原子名称改为N
    for i in range(len(lines)):
        line = lines[i]
        if len(line) >= 52 and line[50:52] == "DU" or line[50:52] == "N3":
            if "N" in line[8:10]:
                lines[i] = line[:50] + "N " + line[52:]

    # 覆盖原文件
    with open(file_path, 'w') as f:
        f.writelines(lines)
        
def process_mol2_with_args(mol2_value):
    # Construct the file name based on the argument
    file_name = f"{mol2_value}_1.mol2"

    # Get the full file path
    folder_path = 'RESP'  # Assuming RESP folder is in the current working directory
    file_path = os.path.join(folder_path, file_name)

    # Process the mol2 file
    process_mol2_file(file_path)
    
    # 读取并打印mol2文件的内容
    with open(file_path, 'r') as mol2_new_print:
        mol2_new_content = mol2_new_print.read()
        print(mol2_new_content)
        
def generate_top(resp_folder):
    # 遍历RESP文件夹中的mol2文件
    for mol2_file in os.listdir(resp_folder):
        if mol2_file.endswith('.mol2'):
            # 提取文件名前三个字符作为res对象
            res = mol2_file[:3]
            global res_rtp
            res_rtp=res

            # 创建存放结果的文件夹
            output_folder = f'./{res}_gromacs_prm'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # 执行parmchk2命令生成.mod文件
            mod_file = f'{res}.mod'
            os.system(f'parmchk2 -i {os.path.join(resp_folder, mol2_file)} -f mol2 -o {mod_file}')

            # 生成leapin文件
            leapin_filename = f'{res}_leap.in'
            with open(leapin_filename, 'w+') as leapin:
                leapin.write(f'source /home/wendao/micromamba/envs/peprobe/dat/leap/cmd/leaprc.protein.ff19SB\n')
                leapin.write(f'loadamberparams {mod_file}\n')
                leapin.write(f'mol=loadmol2 {os.path.join(resp_folder, mol2_file)}\n')
                leapin.write(f'check mol\n')
                leapin.write(f'saveamberparm mol {res}.prm {res}.crd\n')
                leapin.write('quit\n')

            # 调用tleap
            os.system(f'tleap -f {leapin_filename}')

            # 移动生成的文件到指定文件夹
            generated_files = [f'{res}.prm', f'{res}.crd', mod_file, leapin_filename, 'leap.log']
            for file in generated_files:
                if os.path.exists(file):
                    shutil.move(file, os.path.join(output_folder, file))

            # 切换到输出文件夹进行ACpype操作
            os.chdir(output_folder)
            os.system(f'acpype -p {res}.prm -x {res}.crd -c user -o gmx -a amber')

            # 返回到原始工作目录
            os.chdir('..')

            # 移动生成的GROMACS文件到指定文件夹
            gromacs_files = ['MOL_GMX.gro', 'MOL_GMX.top']
            for file in gromacs_files:
                if os.path.exists(file):
                    shutil.move(file, os.path.join(output_folder, f'{res}.{file.split("_")[-1]}'))

            # 清理中间文件
            os.system(f'rm qout QOUT punch md.mdp esout em.mdp')

    print(f'\nFZ-wang reminds you: The GROMACS top files have been generated in the folder "gromacs_prm"!\n')
    
def generate_rtp(resp_file):
    res = res_rtp  # 残基的名称或标识
    ignore_atoms = ["ACE", "NME", "linker"]  # 不需要包含在 .rtp 文件中的原子名称列表
    # 打开并读取 .top 文件
    with open(resp_file, 'r') as f_top:
        top = f_top.readlines()

    # 确定各个部分的起始和结束行索引
    start_atom = top.index('[ atoms ]\n') + 2
    start_bond = top.index('[ bonds ]\n') + 2
    end_atom = top.index('[ bonds ]\n') - 1
    end_bond = top.index('[ pairs ]\n') - 1
    start_angle = top.index('[ angles ]\n') + 2
    end_angle = top.index('[ dihedrals ] ; propers\n') - 1
    start_dihedral = top.index('[ dihedrals ] ; propers\n') + 3
    end_dihedral = top.index('[ dihedrals ] ; impropers\n') - 1
    start_improper = top.index('[ dihedrals ] ; impropers\n') + 3
    end_improper = top.index('[ system ]\n') - 1

    # 初始化 RTP 列表
    rtp_list = []
    rtp_list.append(f'[ {res} ]\n')  # 残基条目
    rtp_list.append(' [ atoms ]\n')  # atoms

    # 定义 include_ffparm 函数，用于排除不需要的行
    def include_ffparm(atom_nums, atom_names):
        for i in atom_nums:
            if int(i) < 13:
                return False
        for j in atom_names:
            if j in ignore_atoms:
                return False
        return True
    
    # 处理 atoms 项
    for i in range(start_atom, end_atom):
        line = top[i]
        atom = list(filter(None, line.replace('\n', '').split()))

        atom_nums, atom_names = [atom[0]], [atom[4]]
        atom_name, atom_type, atom_charge, atom_num = atom[4], atom[1], atom[6], int(atom[0])-12
        
        if include_ffparm(atom_nums, atom_names):
#           print(f'processing atom {atom_name}')
            rtp_list.append(f'    {atom_name:>4}   {atom_type:>2}    {atom_charge:>9}    {atom_num:>2}\n')

    rtp_list.append('\n [ bonds ]\n')

    # 处理 bonds 项
    for j in range(start_bond, end_bond):
        line = top[j]
        bond = list(filter(None, line.replace('\n', '').split()))

        atom_nums, atom_names = bond[0:2], [bond[-3], bond[-1]]
        if include_ffparm(atom_nums, atom_names):
#           print(f'processing bond {bond[-3]}-{bond[-1]}')
            rtp_list.append(f'    {bond[-3]:>4}   {bond[-1]:<4}  {bond[3]}    {bond[4]}\n')

    rtp_list.append(f'    {"-C":>4}   {"N":<4}  1.3790e-01    3.5782e+05\n')
    rtp_list.append('\n [ angles ]\n')

    # 处理 angles 项
    for k in range(start_angle, end_angle):
        line = top[k]
        angle = list(filter(None, line.replace('\n', '').split()))

        atom_nums, atom_names = angle[0:3], [angle[-5], angle[-3], angle[-1]]
        if include_ffparm(atom_nums, atom_names):
#           print(f'processing angle {angle[-5]}-{angle[-3]}-{angle[-1]}')
            rtp_list.append(f'    {angle[-5]:>4}   {angle[-3]:>4}    {angle[-1]:<4}  {angle[4]}   {angle[5]}\n')

    rtp_list.append('\n [ dihedrals ] ; propers\n')

    # 处理 dihedrals 项
    for l in range(start_dihedral, end_dihedral):
        line = top[l].replace('-', ' ')
        dihedral = list(filter(None, line.replace('\n', '').split()))

        atom_nums, atom_names = dihedral[0:4], [dihedral[-4], dihedral[-3], dihedral[-2], dihedral[-1]]
        if include_ffparm(atom_nums, atom_names):
#           print(f'processing dihedrals proper {dihedral[-4]}-{dihedral[-3]}-{dihedral[-2]}-{dihedral[-1]}')
            rtp_list.append(f'    {dihedral[-4]:>4}   {dihedral[-3]:>4}   {dihedral[-2]:>4}   {dihedral[-1]:<4}  {dihedral[5]:>6}   {dihedral[6]:>8}   {dihedral[7]}\n')

    rtp_list.append('\n [ dihedrals ] ; impropers\n')

    # 处理 impropers 项
    for m in range(start_improper, end_improper):
        line = top[m].replace('-', ' ')
        improper = list(filter(None, line.replace('\n', '').split()))

        atom_nums, atom_names = improper[0:4], [improper[-4], improper[-3], improper[-2], improper[-1]]
        if include_ffparm(atom_nums, atom_names):
#           print(f'processing dihedrals impropers {improper[-4]}-{improper[-3]}-{improper[-2]}-{improper[-1]}')
            rtp_list.append(f'    {improper[-4]:>4}   {improper[-3]:>4}   {improper[-2]:>4}   {improper[-1]:<4}  {improper[5]:>6}   {improper[6]:>8}   {improper[7]}\n')

    rtp_list.append('    -C    CA     N     H  180.00   4.60240   2\n    CA    +N     C     O  180.00   4.60240   2\n')

    # 写入生成的 .rtp 文件
    with open(f'{res_rtp}.rtp', 'w') as f_rtp:
        for line in rtp_list:
            f_rtp.write(line)

    # 读取并打印rtp文件的内容
    with open(f'{res_rtp}.rtp', 'r') as rtp_print:
        rtp_content = rtp_print.read()
        print(rtp_content)
        
# 计算Chi角数，并分而治之，设置其对应的初始构象数
def count_rotatable_bonds(smiles_file, cut_off):
    with open(smiles_file, 'r') as file:
        smiles_list = [line.strip() for line in file.readlines()]

    results = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"SMILES字符串 '{smiles}' 无法解析")
            continue

        num_rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        if num_rotatable_bonds==2 and cut_off==10000:
            v_value=10000
            c_value=10000
        elif num_rotatable_bonds==3 and cut_off==10000:
            v_value=10000
            c_value=10000
        elif num_rotatable_bonds==4 and cut_off==10000:
            v_value=10000
            c_value=20000
        elif num_rotatable_bonds==5 and cut_off==10000:
            v_value=10000
            c_value=20000
        elif num_rotatable_bonds==6 and cut_off==10000:
            v_value=10000
            c_value=20000
        elif num_rotatable_bonds==7 and cut_off==10000:
            v_value=10000
            c_value=30000
        elif num_rotatable_bonds==8 and cut_off==10000:
            v_value=10000
            c_value=30000
        elif num_rotatable_bonds==9 and cut_off==10000:
            v_value=10000
            c_value=40000
        else:
            v_value=cut_off
            c_value=50000
        results.append((smiles, num_rotatable_bonds, c_value, v_value))

    return results

# 生成封端后的SMILES字符串
def gen_smiles(input_file, names_list):

    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]

    if len(names_list) != len(smiles_list):
        raise ValueError("The number of provided names does not match the number of SMILES strings.")

    processed_smiles_list = []
    for smiles in smiles_list:
        try:

            processed_smiles = process_smiles(smiles)
            processed_smiles_list.append(processed_smiles)
        except ValueError as e:
            print(e)
            continue
    return processed_smiles_list
    
# 从 dftd4 输出文件中提取能量值
def extract_energy_from_output(out_filename):
    with open(out_filename, 'r') as out_file:
        lines = out_file.readlines()
        for line in lines:
            if "Dispersion energy:" in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # 直接提取倒数第二个元素作为能量值
                        energy = float(parts[-2])
                        return energy
                    except ValueError:
                        print(f"无法将 '{parts[-2]}' 转换为浮点数")
                        return None
    print(f"未找到能量值行在 '{out_filename}' 中")
    return None

# 检查函数，用于检查RDKit生成的构象是否出现了原子重叠
'''实际应用中发现，RDKit对某些集团，例如杂环，羧基等，有时创建的构象会存在扭曲，导致原子重叠。该函数的目的便是筛去这些重叠构象。此处设置的原子间距离阈值为0.6埃，小于该值则判定为重叠，被排除。然而实际应用中可以根据需要进行调整，个人认为0.8以内的任何值都是合理的。如果明确自己的分子不会在建库时出现扭曲现象，则可以去掉该函数，因为该函数时间复杂度较高，会较大影响运行效率'''
def check_for_clashes(mol, threshold=0.6):

    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        pos_i = conf.GetAtomPosition(i)
        for j in range(i + 1, num_atoms):
            pos_j = conf.GetAtomPosition(j)
            distance = pos_i.Distance(pos_j)
            # 检查距离是否小于阈值
            if distance < threshold:
                return True
    return False

# 将SMILES转化为PDB。该函数将被循环调用以创建初始旋转体文库
def smiles_to_pdb_rotamers(smiles_list, names_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(names_list) < len(smiles_list):
        names_list.extend(['UAA'] * (len(smiles_list) - len(names_list)))
    elif len(names_list) > len(smiles_list):
        raise ValueError("The number of provided names exceeds the number of SMILES strings.")

    generated_files = []
    for i, (smiles, name) in enumerate(zip(smiles_list, names_list)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"SMILES字符串 '{smiles}' 无法解析")
            continue
        mol = Chem.AddHs(mol)

        # 添加随机种子，保证每次创建的构象具有随机性
        random_seed = random.randint(1, 1000000)
        params = AllChem.ETKDG()
        params.randomSeed = random_seed
        AllChem.EmbedMolecule(mol, params)
        
        # 使用UFF力场进行结构优化
        ff = UFFGetMoleculeForceField(mol)
        ff.Initialize()

        ff.Minimize()

        # 检查是否存在原子重叠
        if check_for_clashes(mol):
            print(f"构象 {name}_{i+1} 出现原子重叠，跳过该构象")
            continue
            
        # 优化后设置二面角
        set_dihedral_angle(mol, [1, 3, 4, 5], -150.0)
        set_dihedral_angle(mol, [3, 4, 5, 7], 150.0)
        set_dihedral_angle(mol, [2, 1, 3, 4], 4.18)
        set_dihedral_angle(mol, [6, 5, 7, 8], 1.17)
                
        # 将分子保存为 XYZ 文件
        xyz_filename = os.path.join(output_dir, f'{name}_{i+1}.xyz')
        Chem.MolToXYZFile(mol, xyz_filename)

        # 使用 dftd4 进行能量计算
        out_filename = os.path.join(output_dir, f'{name}_{i+1}.out')
        command = f'dftd4 {xyz_filename} -f b3-lyp > {out_filename}'
        result = subprocess.run(command, shell=True, text=True, capture_output=True)

        # 提取能量值
        energy = extract_energy_from_output(out_filename)

        # 生成PDB文件名
        pdb_filename = os.path.join(output_dir, f'{name}_{i+1}.pdb')
        
        # 写入PDB文件并添加能量和SMILES信息
        with open(pdb_filename, 'w') as pdb_file:
            pdb_file.write(f"REMARK SMILES: {smiles}\n")
            pdb_file.write(f"REMARK Energy after optimization: {energy*627.5095:.2f} kcal/mol\n" if energy else "REMARK Energy could not be determined\n")
            pdb_file.write(Chem.rdmolfiles.MolToPDBBlock(mol))
        
        generated_files.append(pdb_filename)

    return generated_files

# 将smiles转化为pdb
def gen_conformers(processed_smiles_list, names_list):
    output_dir = 'pdb_files'
    
    # 清空pdb_files目录中的旧文件
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
    
    generated_files = smiles_to_pdb_rotamers(processed_smiles_list, names_list, output_dir)
    
    # 如果没有生成新的PDB文件，则返回False
    return bool(generated_files)

# 将创建好的pdb文件添加到合并构象库中
def append_pdb_to_combined_file(pdb_file, combined_file):
    # 检查文件名是否是 temp.pdb，如果是则跳过
    if os.path.basename(pdb_file) == 'temp.pdb':
        print(f"Skipping {pdb_file} as it is temp.pdb.")
        return

    # 如果不是 temp.pdb，则将内容添加到 combined_file 中
    with open(pdb_file, 'r') as f_pdb:
        pdb_content = f_pdb.read()
    
    with open(combined_file, 'a') as f_combined:
        f_combined.write(pdb_content)
    
# 对PDB文件中的原子编号进行重新排序
def process_pdb_file_rotamers(filepath, n_value, output_filepath, energy_value):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # 添加能量值注释行
    energy_line = f"REMARK Energy after optimization: {energy_value:.2f}\n"

    # 过滤掉不是以 "HETATM" 或 "ATOM  " 开头的行
    lines = [line for line in lines if line.startswith("HETATM") or line.startswith("ATOM  ")]

    # 字典存储需要重新排序的行
    line_dict = {"C2": None, "O1": None, "C1": None, "H1": None, "H2": None, "H3": None,
                 "N2": None, "C5": None, "H6": None, "H7": None, "H8": None, "H9": None,
                 "N1": None, "C3": None, "C4": None, "O2": None, "H4": None, "H5": None}
    other_lines = []

    for line in lines:
        atom_name = line[12:16].strip()
        if atom_name in line_dict:
            line_dict[atom_name] = line
        else:
            other_lines.append(line)

    # 按指定顺序重新排列行
    reordered_lines = []
    for key in ["C2", "O1", "C1", "H1", "H2", "H3", "N2", "C5", "H6", "H7", "H8", "H9", "N1", "C3", "C4", "O2"]:
        if line_dict[key] is not None:
            reordered_lines.append(line_dict[key])

    reordered_lines.extend(other_lines)

    if line_dict["H4"] is not None:
        reordered_lines.append(line_dict["H4"])

    if line_dict["H5"] is not None:
        reordered_lines.append(line_dict["H5"])

    # 修改残基名
    for i in range(len(reordered_lines)):
        if i < 6:
            reordered_lines[i] = reordered_lines[i][:17] + "ACE" + reordered_lines[i][20:]
        elif i < 12:
            reordered_lines[i] = reordered_lines[i][:17] + "NME" + reordered_lines[i][20:]
        else:
            reordered_lines[i] = reordered_lines[i][:17] + n_value + reordered_lines[i][20:]

    # 在开头添加能量值注释行
    reordered_lines.insert(0, energy_line)

    # 添加 "END" 行
    reordered_lines.append("END\n")

    # 写回新文件
    with open(output_filepath, 'w') as file:
        file.writelines(reordered_lines)

# 能量提取函数
def extract_energy_from_pdb(pdb_content):
    lines = pdb_content.splitlines()
    energy = None
    for line in lines:
        if line.startswith("REMARK Energy after optimization:"):
            # 改进正则表达式以匹配正负号
            match = re.search(r'[-+]?\d*\.\d+', line)
            if match:
                energy = float(match.group())
            break
    if energy is None:
        raise ValueError("No energy information found in PDB content")
    return energy
  
# 拆分合并的PDB文件并按能量值由小到大排序
def split_combined_pdb(combined_pdb_file, output_dir, n_value):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdb_files = []
    energies = {}

    with open(combined_pdb_file, 'r') as f:
        content = f.read().strip().split('END\n')

    for idx, pdb_block in enumerate(content):
        if pdb_block.strip():
            pdb_content = pdb_block.strip() + '\nEND'
            energy = extract_energy_from_pdb(pdb_content)

            # 创建PDB文件名，使用能量值命名
            pdb_filename = os.path.join(output_dir, f'conformer_{energy:.2f}.pdb')

            with open(pdb_filename, 'w') as pdb_out:
                pdb_out.write(pdb_content)

            pdb_files.append((pdb_filename, energy))
            energies[pdb_filename] = energy

            # 对每个拆分的PDB文件的原子编号进行重新排序
            output_filepath = f"{pdb_filename}_processed.pdb"
            process_pdb_file_rotamers(pdb_filename, n_value, output_filepath, energy)
            print(f"已处理PDB文件: {output_filepath}")

    # 按能量排序生成的PDB文件
    sorted_files = sorted(pdb_files, key=lambda x: x[1])

    # 创建新的合并PDB文件并按能量顺序写入排序后的pdb
    combined_sorted_file = 'combined_sorted_pdb_files.pdb'
    with open(combined_sorted_file, 'w') as f_combined:
        for pdb_file, _ in sorted_files:
            with open(pdb_file + "_processed.pdb", 'r') as f_pdb:
                pdb_content = f_pdb.read()
                f_combined.write(pdb_content)

    return combined_sorted_file

# 定义主链原子名
backbone_atoms = {'C3', 'C4', 'O1', 'N1', 'O2'}
    
# 获取主链原子
def get_backbone_atoms(structure):
    return [atom for atom in structure.get_atoms() if atom.get_name() in backbone_atoms]

# 按主链对所有pdb进行align，并使用对齐后的pdb文件并覆盖原始PDB文件
def align_and_overwrite(ref_structure, pdb_file, parser):
    structure = parser.get_structure(pdb_file, pdb_file)
    
    # 获取主链原子
    ref_atoms = get_backbone_atoms(ref_structure)
    mobile_atoms = get_backbone_atoms(structure)
    
    # 使用 Superimposer 进行对齐
    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, mobile_atoms)
    super_imposer.apply(structure.get_atoms())

    # 保存对齐后的结构到原始PDB文件中
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)

# RMSD计算函数
def calculate_rmsd(P, Q):

    diff = P - Q
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd


# 从PDB文件中提取排除指定原子外的坐标和名称
def get_coordinates(structure, atoms_to_exclude):
    atoms = []
    coords = []
    for atom in structure.get_atoms():
        if atom.get_name() not in atoms_to_exclude and 'H' not in atom.get_name():  # 排除主链，封端和所有氢原子
            atoms.append(atom.get_name())
            coords.append(atom.get_coord())
    
    return np.array(coords), atoms

# 计算两个构象之间的RMSD
def align_and_calculate_rmsd(ref_structure, temp_structure, original_pdb_file):
    # 指定在RMSD计算中要排除的主链，封端和所有氢原子
    atoms_to_exclude = {'C1', 'C2', 'C4', 'C5', 'O1', 'O2', 'N1', 'N2'}
    
    coords1, atom_names1 = get_coordinates(ref_structure, atoms_to_exclude)
    coords2, atom_names2 = get_coordinates(temp_structure, atoms_to_exclude)
    
    if coords1.shape != coords2.shape:
        raise ValueError("两个PDB文件的坐标数量不一致")
    
    rmsd_value = calculate_rmsd(coords1, coords2)
    
    return rmsd_value

# 遍历每一个构象，将其与最终构象库中的所有构象进行RMSD比对，当且仅当被遍历的构象与最终构象库中的所有构象的RMSD值均大于阈值时，将其写入最终的合并文件
def align_structures_and_check_rmsd(pdb_files, parser, aligned_combined_pdb_filename, rmsd_threshold):
    aligned_structures = []
    conformer_count = 0

    with open(aligned_combined_pdb_filename, 'w') as aligned_combined_pdb, open('prechi_structure.pdb', 'w') as prechi_structure:
        ref_pdb_file = pdb_files[0]
        ref_structure = parser.get_structure('reference', ref_pdb_file)
        align_and_overwrite(ref_structure, ref_pdb_file, parser)
        aligned_structures.append(ref_structure)

        for pdb_file in pdb_files[1:]:
            align_and_overwrite(ref_structure, pdb_file, parser)
            temp_structure = parser.get_structure('temp_aligned', pdb_file)

            # 将对齐后的结构写入 prechi_structure.pdb 文件
            with open(pdb_file, 'r') as temp_file:
                lines = temp_file.readlines()
                for line in lines:
                    if line.startswith(('REMARK', 'ATOM', 'HETATM', 'END')):
                        prechi_structure.write(line)

            if conformer_count >= v_value:
                continue

            # 比较 temp_structure 与 aligned_combined_pdb_files.pdb 中所有结构的 RMSD 值
            should_keep = True
            for aligned_structure, aligned_pdb_file in reversed(list(zip(aligned_structures, pdb_files))):
                rmsd = align_and_calculate_rmsd(aligned_structure, temp_structure, aligned_pdb_file)
                print(f"RMSD between {os.path.basename(pdb_file)} and {os.path.basename(aligned_pdb_file)}: {rmsd}")
                if rmsd < rmsd_threshold:
                    should_keep = False
                    break
            
            # 如果不存在 RMSD 小于阈值的结果，则将 temp_structure 写入最终的合并文件
            if should_keep:
                with open(pdb_file, 'r') as temp_file:
                    lines = temp_file.readlines()
                    for line in lines:
                        if line.startswith(('REMARK', 'ATOM', 'HETATM', 'END')):
                            aligned_combined_pdb.write(line)
                
                conformer_count += 1
                aligned_structures.append(temp_structure)

    # 输出实际生成的构象个数
    print(f"实际生成的构象个数为: {conformer_count}")

# 从 combined_sorted_pdb_files.pdb 文件中读取所有 PDB 结构并进行 RMSD 筛选
def screen_rmsd(rmsd_threshold):
    parser = PDBParser(QUIET=True)

    # 读取 combined_pdb_files.pdb 文件
    with open('combined_sorted_pdb_files.pdb', 'r') as f:
        content = f.read().strip().split('END\n')

    # 将每个 PDB 结构写入单独的文件
    pdb_files = []
    for i, pdb_content in enumerate(content):
        if pdb_content.strip():
            pdb_filename = f'pdb_{i}.pdb'
            with open(pdb_filename, 'w') as pdb_file:
                pdb_file.write(pdb_content + 'END\n')
            pdb_files.append(pdb_filename)

    # 创建一个新的文件来存储所有对齐后的 PDB 结构
    aligned_combined_pdb_filename = 'aligned_combined_pdb_files.pdb'

    align_structures_and_check_rmsd(pdb_files, parser, aligned_combined_pdb_filename, rmsd_threshold)

    # 删除中间生成的 PDB 文件
    for pdb_file in pdb_files:
        os.remove(pdb_file)

    print(f"已创建合并 PDB 文件: aligned_combined_pdb_files.pdb")

# pdb读入函数
def read_pdb_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

# pdb写入函数
def write_pdb_file(file_path, lines):
    with open(file_path, 'w') as file:
        file.writelines(lines)

# 清洗函数
def remove_ace_nme(lines):
    return [line for line in lines if "ACE" not in line and "NME" not in line]

# 以END行为分隔符对合并构象库进行分隔
def split_pdb_by_end(lines):
    pdb_blocks = []
    block = []
    for line in lines:
        if line.strip() == "END":
            block.append(line)
            pdb_blocks.append(block)
            block = []
        else:
            block.append(line)
    if block:
        pdb_blocks.append(block)
    return pdb_blocks

# 读取params中的原子定义行与Chi角定义行
def parse_params_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    atom_lines = [line for line in lines if line.startswith("ATOM")]
    chi_lines = [line[5:].strip() for line in lines if line.startswith("CHI")]
    chi_lines = [line for line in chi_lines if 'H' not in line]
    return atom_lines, chi_lines

# 创建一一对应键值对，用于进行二面角筛选
def create_atom_mapping(list1, list2):
    atom_mapping = {}
    for atom1, atom2 in zip(list1, list2):
        key = atom1[12:17].strip()
        value = atom2[4:9].strip()
        atom_mapping[key] = value
    return atom_mapping

# 使用atom_map识别二面角
def replace_chi_atoms(chi_lines, atom_mapping):
    reversed_mapping = {v: k for k, v in atom_mapping.items()}
    replaced_chi_lines = []
    for chi_line in chi_lines:
        atoms = chi_line.split()
        replaced_atoms = []
        for atom in atoms:
            if atom.strip() in reversed_mapping:
                replaced_atoms.append(reversed_mapping[atom.strip()])
            else:
                replaced_atoms.append(atom.strip())
        replaced_chi_lines.append(' '.join(replaced_atoms))
    return replaced_chi_lines

# 二面角计算函数，用于计算单个二面角的值
def calculate_dihedral(coords1, coords2, coords3, coords4):
    b1 = coords2 - coords1
    b2 = coords3 - coords2
    b3 = coords4 - coords3
    
    b2 /= np.linalg.norm(b2)
    
    v = b1 - np.dot(b1, b2) * b2
    w = b3 - np.dot(b3, b2) * b2
    
    x = np.dot(v, w)
    y = np.dot(np.cross(b2, v), w)
    
    angle = np.degrees(np.arctan2(y, x))
    
    if angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle
    
    return angle

# 坐标读取函数
def find_atom_coordinates(pdb_block, atom_name):
    for line in pdb_block:
        if line[12:17].strip() == atom_name:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            return np.array([x, y, z])
    return None

# 读取合每个pdb文件的所有二面角
def calculate_all_dihedrals(pdb_block, chi_lines, atom_mapping):
    dihedral_angles = []
    for chi_line in chi_lines:
        atoms = chi_line.split()
        coords = [find_atom_coordinates(pdb_block, atom) for atom in atoms]
        coords = [coord for coord in coords if coord is not None]
        if len(coords) == 4:
            angle = calculate_dihedral(*coords)
            dihedral_angles.append(angle)
    return dihedral_angles

# 二面角差值计算
def compare_dihedrals(reference_angles, pdb_angles):
    keep = False
    for ref_angle, pdb_angle in zip(reference_angles, pdb_angles):
        diff = abs(ref_angle - pdb_angle)
#        print(f"Reference angle: {ref_angle:.2f}, PDB angle: {pdb_angle:.2f}, Difference: {diff:.2f}")
        if diff >= 5:
#            print("Keep")
            keep = True
            break
#    if not keep:
#        print("Not keep")
    return keep

# 清理中间文件
def delete_temp_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
#            print(f"Deleted {file_path}")

# Chi角搜索部分主函数，用于进行Chi角检索
def Chi_screen(args_params):
    combined_file_path = 'prechi_structure.pdb'
    chi_screen_file_path = 'Chi_screen.pdb'
    params_file_path = args_params  # Replace with actual params file path

    temp_files = []

    # 清空 Chi_screen.pdb 文件
    open(chi_screen_file_path, 'w').close()

    pdb_lines = read_pdb_file(combined_file_path)
    pdb_lines = remove_ace_nme(pdb_lines)
    preprocessed_pdb_path = 'preprocessed_pdb_files.pdb'
    write_pdb_file(preprocessed_pdb_path, pdb_lines)
    temp_files.append(preprocessed_pdb_path)
    
    pdb_blocks = split_pdb_by_end(pdb_lines)
    
    for i, block in enumerate(pdb_blocks):
        block_file_path = f"pdb_block_{i+1}.pdb"
        write_pdb_file(block_file_path, block)
        temp_files.append(block_file_path)
    
    # Step 1: Save the first PDB block to Chi_screen.pdb
    write_pdb_file(chi_screen_file_path, pdb_blocks[0])
    chi_screen_blocks = split_pdb_by_end(read_pdb_file(chi_screen_file_path))

    atom_lines, chi_lines = parse_params_file(params_file_path)

    i = 0
    j = 1
    for pdb_block in pdb_blocks[1:]:
        i += 1
        atom_lines_in_pdb = [line for line in pdb_block if line.startswith("HETATM") or line.startswith("ATOM")]
        atom_mapping = create_atom_mapping(atom_lines_in_pdb, atom_lines)
        replaced_chi_lines = replace_chi_atoms(chi_lines, atom_mapping)
        
        reference_angles = calculate_all_dihedrals(pdb_block, replaced_chi_lines, atom_mapping)
#        print('reference:')
#        print(reference_angles)
        
        should_keep = True
        
        for chi_screen_block in reversed(chi_screen_blocks):
            pdb_angles = calculate_all_dihedrals(chi_screen_block, replaced_chi_lines, atom_mapping)
        #    print('compare:')
        #    print(pdb_angles)
            if not compare_dihedrals(reference_angles, pdb_angles):
                should_keep = False
                break
        
        if should_keep:
            j+=1
            with open(chi_screen_file_path, 'a') as f:
                f.writelines(pdb_block)
            # Add new block to chi_screen_blocks for future comparisons
            chi_screen_blocks.append(pdb_block)
            print(f"Keep conformer No.{i}")
#        else:
#            print("Not keeping this conformer.")
#        print(f'number:{i} conformer has been screened')
    print(f'The number of PDBs that meet the Chi angle requirements:{j}')
    delete_temp_files(temp_files)

# 计算两个构象之间的RMSD
def calculate_chi_aligned_rmsd(ref_structure, temp_structure):
    # 排除主链
    atoms_to_exclude = {'C1', 'C2', 'C4', 'C5', 'O1', 'O2', 'N1', 'N2'}
    
    coords1, atom_names1 = get_coordinates(ref_structure, atoms_to_exclude)
    coords2, atom_names2 = get_coordinates(temp_structure, atoms_to_exclude)
    
    if coords1.shape != coords2.shape:
        raise ValueError("两个PDB文件的坐标数量不一致")
    
    rmsd_value = calculate_rmsd(coords1, coords2)
    
    return rmsd_value

# 文件读入函数
def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

# 文件写入函数
def write_file(file_path, lines):
    with open(file_path, 'w') as file:
        file.writelines(lines)

# 清洗函数，去除无关行
def process_chi_screen(file_path):
    lines = read_file(file_path)
    filtered_lines = [line for line in lines if line.startswith(('HETATM', 'ATOM  ', 'END'))]
    write_file(file_path, filtered_lines)

# 清洗函数，去除封端
def process_aligned_combined(file_path):
    lines = read_file(file_path)
    filtered_lines = [line for line in lines if 'ACE' not in line and 'NME' not in line]
    write_file(file_path, filtered_lines)

# 构象提取函数
def extract_conformers(lines):
    blocks = []
    block = []
    for line in lines:
        if line.strip() == "END":
            block.append(line)
            blocks.append(''.join(block))
            block = []
        else:
            block.append(line)
    if block:
        blocks.append(''.join(block))
    return blocks

# 坐标提取函数
def extract_coordinates_from_block(pdb_block):
    """
    从 PDB 文件的文本块中提取原子坐标。假设 PDB 块格式正确，
    每行代表一个原子，并且坐标位于标准的列范围内。
    返回一个坐标列表 [(x1, y1, z1), (x2, y2, z2), ...]
    """
    coordinates = []
    for line in pdb_block.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # 从 PDB 的第31到54列提取 x, y, z 坐标
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            coordinates.append((x, y, z))
    return coordinates

# 比较函数，用于检查两个构象是否为同一构象
def coordinates_are_equal(block1, block2):

    coords1 = extract_coordinates_from_block(block1)
    coords2 = extract_coordinates_from_block(block2)

    # 如果原子数量不同，直接返回 False
    if len(coords1) != len(coords2):
        return False

    # 比较每个原子的坐标是否完全相同
    for atom1, atom2 in zip(coords1, coords2):
        if atom1 == atom2:
            print(f'{atom1}={atom2}')
        if atom1 != atom2:
            print('FALSE')
            return False
    
    return True  # 如果所有原子坐标都相同，返回 True

# 合并函数，用于将Chi角搜索和RMSD搜索得到的两个构象库合并，去除相同构象
def append_unique_conformers(chi_screen_path, aligned_combined_path):
    chi_screen_lines = read_file(chi_screen_path)
    aligned_combined_lines = read_file(aligned_combined_path)
    
    new_lines = []
    new_conformers_count = 0

    chi_blocks = extract_conformers(chi_screen_lines)
    aligned_blocks = extract_conformers(aligned_combined_lines)
    
    for chi_block in chi_blocks:
        is_unique = True
        
        for aligned_block in aligned_blocks:
            # 比较原子坐标是否完全相同
            if coordinates_are_equal(chi_block, aligned_block):
                is_unique = False
                break
        
        if is_unique:
            new_lines.append(chi_block)
            new_conformers_count += 1
    
    if new_lines:
        with open(aligned_combined_path, 'a') as file:
            file.writelines(new_lines)
    
    print(f"Number of new conformers added: {new_conformers_count}")

# 定义分隔文件夹
split_pdb_directory = 'split_pdbs'

# 从PDB文件中删除包含'ACE'或'NME'的行，并重新编写原子编号
def remove_ace_nme_lines(pdb_filename):

    lines = []
    current_residue_number = 1

    with open(pdb_filename, 'r') as f:
        for line in f:
            if 'ACE' in line or 'NME' in line:
                continue
            if line.startswith('ATOM'):
                # 重新编写原子编号
                line = f"{line[:6]}{current_residue_number:>4}{line[10:]}"
                current_residue_number += 1
            lines.append(line)

    # 将修改后的行写回原始PDB文件
    with open(pdb_filename, 'w') as f:
        f.write(''.join(lines))

# 将PDB文件拆分为单个构象
def split_pdb_file(input_pdb_file):

    output_directory = 'split_pdbs'
    os.makedirs(output_directory, exist_ok=True)

    conformation_index = 1
    pdb_content = []

    with open(input_pdb_file, 'r') as f:
        for line in f:
            if line.strip() == 'END':
                pdb_filename = os.path.join(output_directory, f'conformation_{conformation_index}.pdb')
                with open(pdb_filename, 'w') as pdb_file:
                    pdb_file.write('\n'.join(pdb_content) + '\nENDMDL\n')

                conformation_index += 1
                pdb_content = []
            else:
                pdb_content.append(line.strip())

    if pdb_content:
        pdb_filename = os.path.join(output_directory, f'conformation_{conformation_index}.pdb')
        with open(pdb_filename, 'w') as pdb_file:
            pdb_file.write('\n'.join(pdb_content) + '\nENDMDL\n')

# 读取XXX_temps.params文件并替换PDB文件中的[12:16]为非天然氨基酸名称
def process_all_pdb_files(directory, non_natural_aa_codes):

    for filename in os.listdir(directory):
        if filename.endswith('.pdb'):
            pdb_filepath = os.path.join(directory, filename)
            remove_ace_nme_lines(pdb_filepath)
            # 读取PDB文件并替换[12:16]为非天然氨基酸名称
            lines = []
            with open(pdb_filepath, 'r') as f:
                for idx, line in enumerate(f):
                    if line.strip() == 'ENDMDL':
                        lines.append(line)
                        continue
                    if idx < len(non_natural_aa_codes):
                        code = non_natural_aa_codes[idx]
                        line = line[:12] + code + line[16:]
                    lines.append(line)
            
            # 将修改后的行写回原始PDB文件
            with open(pdb_filepath, 'w') as f:
                f.write(''.join(lines))

            print(f"Processed: {filename}")

# 将split_pdb_directory中的所有PDB文件合并为单个输出文件
def combine_pdb_files(output_filename):

    with open(output_filename, 'w') as output_file:
        conformation_index = 1
        for filename in sorted(os.listdir(split_pdb_directory)):
            if filename.endswith('.pdb'):
                pdb_filepath = os.path.join(split_pdb_directory, filename)
                with open(pdb_filepath, 'r') as pdb_file:
                    output_file.write(f"MODEL     {conformation_index}\n")
                    for line in pdb_file:
                        output_file.write(line)
                    conformation_index += 1

# 删除split_pdb_directory
def remove_split_pdb_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    try:
        os.rmdir(directory)
    except Exception as e:
        print(f"Failed to delete {directory}. Reason: {e}")

# 调整索引号，将第二列（索引10:11）减去12，使其从1开始编号
def adjust_hetatm_records(pdb_filename):

    lines = []
    with open(pdb_filename, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                original_number = int(line[9:11])
                modified_number = original_number - 12
                # Adjust right alignment
                if modified_number < 10:
                    line = f"{line[:9]} {modified_number}{line[11:]}"
                else:
                    line = f"{line[:9]}{modified_number}{line[11:]}"
            lines.append(line)

    with open(pdb_filename, 'w') as f:
        f.writelines(lines)

# 执行函数，调用以上modify功能，对pdb文件进行修改
def modify(non_natural_aa_abbreviation):

    params_filename = f"{non_natural_aa_abbreviation}_temps.params"

    # 读取XXX_temps.params文件中的非天然氨基酸原子编号
    non_natural_aa_codes = []
    with open(params_filename, 'r') as params_file:
        for line in params_file:
            if line.startswith('ATOM'):
                code = line.strip()[5:9]
                non_natural_aa_codes.append(code)

    # 拆分合并的PDB文件
    input_pdb_file = 'aligned_combined_pdb_files.pdb'
    split_pdb_file(input_pdb_file)

    # 处理拆分后的PDB文件
    process_all_pdb_files(split_pdb_directory, non_natural_aa_codes)

    # 合并处理后的PDB文件
    combined_pdb_filename = 'merged_combined_pdb_files.pdb'
    combine_pdb_files(combined_pdb_filename)

    # 调整原子索引号
    adjust_hetatm_records(combined_pdb_filename)

    # 删除split_pdb_directory
    remove_split_pdb_directory(split_pdb_directory)
    print(f"Combined PDB files into '{combined_pdb_filename}' and deleted '{split_pdb_directory}'.")
    
def delete_files(n_value):
    # 需要删除的文件列表
    files_to_delete = [
        'combined_pdb_files.pdb', 
        'combined_sorted_pdb_files.pdb', 
        'aligned_combined_pdb_files.pdb',
        'Chi_screen.pdb',
        'prechi_structure.pdb',
        f'smd_{n_value}.pdb',     # 添加 smd_{n_value}.pdb
        'smd_opt.traj',           # 添加 smd_opt.traj
        'gas_opt.traj',           # 添加 gas_opt.traj
        f'gas_{n_value}.pdb'      # 添加 gas_{n_value}.pdb
    ]
    
    # 删除文件
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Successfully deleted {file}")
        except FileNotFoundError:
            print(f"File {file} not found")
        except PermissionError:
            print(f"Permission denied to delete {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    # 删除文件夹
    directories_to_delete = [
        'pdb_files', 
        'split_pdb_files', 
        'PDB_rearranged', 
        'GJF', 
        'mol', 
        'RESP'
    ]
    
    for directory in directories_to_delete:
        if os.path.exists(directory):
            shutil.rmtree(directory, ignore_errors=True)
            print(f"Deleted '{directory}' directory")
    
    # 删除单个文件
    try:
        os.remove('molecule.chk')
        print("Deleted 'molecule.chk'")
    except FileNotFoundError:
        print("File 'molecule.chk' not found")
    except PermissionError:
        print("Permission denied to delete 'molecule.chk'")
    except Exception as e:
        print(f"Error deleting 'molecule.chk': {e}")
    
    print("Intermediate files and folders deleted.")

def main(input_file, names_list,  rmsd_threshold, cut_off):
    # 确保所有需要的文件夹存在或根据需要创建它们
    required_dirs = ['pdb_files', 'PDB_rearranged', 'mol']
 
    resp_folder = 'RESP'
    output_dir = 'pdb_files'
    output_rearranged_dir = 'PDB_rearranged'
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")
    
    # 计算体系电荷
    system_charge = calculate_system_charge(input_file)
    print(f"System charge: {system_charge}")
    
    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
 
    if len(names_list) != len(smiles_list):
        raise ValueError("The number of provided names does not match the number of SMILES strings.")
    else:
        print("names_list=", names_list)
        print("smiles_list=", smiles_list)

    # 处理SMILES
    processed_smiles_list = []
    for smiles in smiles_list:
        try:
            processed_smiles = process_smiles(smiles)
            processed_smiles_list.append(processed_smiles)
        except ValueError as e:
            print(e)
            continue
 
    # 将处理后的SMILES转换为PDB
    generated_files = smiles_to_pdb(processed_smiles_list, names_list, output_dir)
 
    # 重新排序并处理PDB文件
    process_files(generated_files, names_list, output_rearranged_dir)

    # 将PDB转为mol文件
    process_pdb_files(input_file, 'PDB_rearranged')
    
        # 获取生成的mol文件列表
    mol_files = [f for f in os.listdir('mol') if f.endswith('.mol')]
    
    # 遍历每个mol文件，转化为params文件并存放到对应的文件夹中
    for mol_file in mol_files:
        base_name = os.path.splitext(mol_file)[0]
        name = base_name.split('_')[0]
        mol_filepath = os.path.join('mol', mol_file)
        molfile_to_params(mol_filepath, name)
        molfile_to_params_temps(mol_filepath, name)
    
    # 确保RESP文件夹存在
    if not os.path.exists(resp_folder):
        os.makedirs(resp_folder)
    
    # 生成 RESP 文件
    for filename in os.listdir("PDB_rearranged"):
        if filename.endswith('.pdb'):
            pdb_file = os.path.join("PDB_rearranged", filename)  # 获取 PDB 文件路径
            res = os.path.splitext(filename)[0]  # 获取文件名（不包括扩展名）
            resp = os.path.join(resp_folder, res)  # RESP 文件路径
    
            # 复制 PDB 文件到当前工作目录
            shutil.copyfile(pdb_file, f'{res}.pdb')
    
            # 运行 generate_opt 函数
            generate_opt(f'{res}.pdb', resp,resp_folder, system_charge)
    
            # 将 PDB 文件移动回 RESP 文件夹
            shutil.move(f'{res}.pdb', os.path.join(resp_folder, f'{res}.pdb'))
    
    
    # 用RESP电荷替换params的原始电荷
    params_charge_adjust(resp_folder)
    
    # 计算capcharge
    calculate_capcharge(resp_folder)
    
    n_value = args.names[0]
    # 处理params文件
    process_params_file(n_value)
    
    mol2_filepath = os.path.join('RESP', f'{n_value}_1.mol2')
    temps_filepath = os.path.join(f'{n_value}_temps.params')
 
    # 计算体系电荷
    #system_charge = calculate_system_charge('input.smiles')
    #print(f"System charge: {system_charge}")
 
    # 读取mol2文件的电荷信息
    atom_lines = read_mol2_file(mol2_filepath)
    charges_list = []
    for line in atom_lines:
        parts = line.split()
        charge = float(parts[-1])
        charges_list.append(charge)
    
    # 电荷修正，使其为整数
    rounded_charges = adjust_charges_to_integer(charges_list,system_charge)
    print(f"Charges list after adjustment: {rounded_charges}")
 
    # 对temp_params进行电荷修正和封端加和
    modify_params_file(temps_filepath, rounded_charges)
    print(f"Modified {temps_filepath} with rounded charges.")
 
    # 将更新信息传递给params文件
    params_filepath = os.path.join(f'{n_value}.params')
    update_params_file_with_temps(params_filepath, temps_filepath)
    print(f"Updated {temps_filepath} with data from {params_filepath}")
    
    process_mol2_with_args(n_value)
    
    # 生成top文件
    generate_top(resp_folder)
    
    resp = f"{res_rtp}_gromacs_prm/MOL.amb2gmx/MOL_GMX.top"
 
    # 生成rtp文件
    generate_rtp(resp)

    # 设置 OMP_NUM_THREADS 环境变量为 1，限制 CPU 使用为 1 个 CPU
    '''限制单核运行主要是防止dftd4抢夺别的cpu资源导致效率降低，但有时该设置会导致一些报错，目前原因未知，如果报错可以尝试去掉这一设置'''
    os.environ['OMP_NUM_THREADS'] = '1'
    
    results = count_rotatable_bonds(input_file, cut_off)
    for smiles, num_rotatable_bonds, c_value, v_value in results:
        print(f"SMILES: {smiles}  自由旋转键数目: {num_rotatable_bonds}  c_value: {c_value}  v_value: {v_value}")
        # 将c_value和v_value输出为全局变量
        globals()['c_value'] = c_value
        globals()['v_value'] = v_value
        globals()['num_rotatable_bonds'] = num_rotatable_bonds

    combined_file = 'combined_pdb_files.pdb'

    if os.path.exists(combined_file):
        os.remove(combined_file)
    
    # 生成封端后的SMILES
    processed_smiles_list = gen_smiles(input_file, names_list)
    
    # 循环调用gen_conformers函数，直到生成c_value个构象(10000个)
    for i in range(c_value):
        print(f"Running iteration {i + 1}...")
        success = gen_conformers(processed_smiles_list, names_list)
        
        if not success:
            print(f"No valid conformers generated in iteration {i + 1}, skipping PDB processing.")
            continue
        
        # 将pdb_files文件夹中的所有PDB文件内容追加到combined_pdb_files.pdb中
        for pdb_file in os.listdir('pdb_files'):
            if pdb_file.endswith('.pdb'):
                pdb_filepath = os.path.join('pdb_files', pdb_file)
                append_pdb_to_combined_file(pdb_filepath, combined_file)
        
        print(f"Iteration {i + 1} completed.")
        
    combined_pdb_file = 'combined_pdb_files.pdb'  # 假设这是您的合并PDB文件名
    output_directory = 'split_pdb_files'  # 存放拆分后PDB文件的目录
    try:
        sorted_combined_file = split_combined_pdb(combined_pdb_file, output_directory, n_value)
        print(f"已创建排序后的合并PDB文件: {sorted_combined_file}")

    except Exception as e:
        print(f"错误: {e}")
    
    args_params = f'{n_value}_temps.params'
    
    # 设置RMSD阈值
    if rmsd_threshold==0.001 and num_rotatable_bonds==2:
        rmsd_threshold = 0.01
    elif rmsd_threshold==0.001 and num_rotatable_bonds==3:
        rmsd_threshold = 0.01
    elif rmsd_threshold==0.001 and num_rotatable_bonds==4:
        rmsd_threshold = 0.01
    elif rmsd_threshold==0.001 and num_rotatable_bonds==5:
        rmsd_threshold = 0.01
    elif rmsd_threshold==0.001 and num_rotatable_bonds==6:
        rmsd_threshold = 0.01
    elif rmsd_threshold==0.001 and num_rotatable_bonds==7:
        rmsd_threshold = 0.01
    elif rmsd_threshold==0.001 and num_rotatable_bonds==8:
        rmsd_threshold = 0.01
    elif rmsd_threshold==0.001 and num_rotatable_bonds==9:
        rmsd_threshold = 0.01
    else:
        rmsd_threshold = 0.01
    print(rmsd_threshold)

    # RMSD筛选
    try:
        screen_rmsd(rmsd_threshold)

    except Exception as e:
        print(f"错误: {e}")
    
    # Chi角筛选
    Chi_screen(args_params)

    # 定义筛选后输出的构象库文件
    chi_screen_path = 'Chi_screen.pdb'
    aligned_combined_path = 'aligned_combined_pdb_files.pdb'
    
    # 对构象库进行后处理
    process_chi_screen(chi_screen_path)
    process_aligned_combined(aligned_combined_path)
    
    # 合并两个构象库
    append_unique_conformers(chi_screen_path, aligned_combined_path)
    
    non_natural_aa_abbreviation = n_value

    # 格式化构象库
    try:
        modify(non_natural_aa_abbreviation)

    except Exception as e:
        print(f"Error: {e}")

    # 构建文件名
    filename = f'{n_value}.params'

    # 读取文件内容
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 添加新行
    lines.append('PDB_ROTAMERS merged_combined_pdb_files.pdb\n')

    # 写回文件
    with open(filename, 'w') as file:
        file.writelines(lines)

    print(f'Added line to {filename} successfully.')
    
    delete_files(n_value)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process SMILES and convert to PDB files.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input SMILES file.')
    parser.add_argument('-n', '--names', required=True, nargs='+', help='Names for each SMILES (UAA if not specified).')
    parser.add_argument('-d', '--rmsd_threshold', type=float, default=0.001,
                        help='RMSD threshold for filtering conformers')
    parser.add_argument('-c', '--cut_off', type=int, default=10000,
                        help='Cut off value for the number of conformers')    
    args = parser.parse_args()
    main(args.input, args.names, args.rmsd_threshold, args.cut_off)
