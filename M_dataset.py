import os
import ase
from ase.io import extxyz
import numpy as np
import time
import pickle
import json
from ase.io import Trajectory as T
import numpy as np

ptable = ['null', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Uuq', 'Uup', 'Uuh', 'Uus', 'Uuo']

def get_energy(Atoms):
    try:
        energy = Atoms.info['REF_energy']
    except KeyError:
        try:
            energy = Atoms.info['Energy']
        except KeyError:
            try:
                energy = Atoms.info['energy[eV]']
            except KeyError:
                try:
                    energy = Atoms.get_potential_energy()
                except (AttributeError, RuntimeError): 
                    print('No value for total energy was found')
                    energy = 10
    return energy


def get_forces(Atoms):
    try:
        forces = Atoms.arrays['forceseV/Ang']
    except KeyError:
        try:
            forces = np.array(Atoms.arrays['REF_forces'])
        except KeyError:
            try:
                forces = Atoms.arrays['force']
            except KeyError:
                try:
                    forces = np.array(Atoms.get_forces(apply_constraint=False))
                except (AttributeError, RuntimeError): 
                    print('No value for forces was found')
                    n_atoms = len(Atoms.get_positions())
                    forces = np.zeros((n_atoms, 3))
        
    return forces


def get_positions(Atoms):
    pos = np.array(Atoms.get_positions())
    return pos

def get_cell(Atoms):
    cell = np.array(Atoms.get_cell())
    return cell

def get_pbc(Atoms):
    pbc = Atoms.get_pbc()
    return pbc

def get_symbols(Atoms):
    symbols = str(Atoms.symbols)
    return symbols



def get_atomic_force(forces):
    import math
    atomic_f = 0
    for atom_f in forces:
        atomic_f += math.sqrt(atom_f[0]**2 + atom_f[1]**2 + atom_f[2]**2)
    atomic_f /= len(forces)
    return atomic_f


class Dataset:
    '''
    Convert OC20 files into .npy files.
    '''
    def __init__(self, _path, a, b):
        self.path = _path
        self.a = a
        self.b = b

    def extract_dataset(self):
        anomalies = 0
        for i in range(self.a, self.b):
            dataset = np.array([], dtype=object)
            dataset = np.hstack((dataset, np.array(['index = 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4', 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))

            print(f'structure_{i}', flush = True)
            File = open(f'{i}.extxyz', 'r')
            txt_file = open(f'{i}.txt')
            read = [line for line in txt_file]
            Id_list = [line.split(',')[0].replace('random', '') for line in read]
            start = time.time()
            idx = 0
            while True:
                try:
                    structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
                    symbol = str(structure[0].symbols)
                    atoms = np.array(structure[0].get_chemical_symbols())
                    atoms = np.array( [ptable.index(j) for j in atoms])
                    natom = len(atoms)
                    cell = np.array(structure[0].get_cell())
                    pos = np.array(structure[0].get_positions())
                    energy = structure[0].get_total_energy()
                    force = np.array(structure[0].get_forces(apply_constraint=False))
        #            if _map[f'random{Id_list[idx]}']['anomaly'] == 0:
                    Id = str(i)+'_'+str(idx)+'_'+Id_list[idx]
                    lst = [idx, natom, atoms, cell, symbol, pos, energy, force, Id]
                    dataset = np.vstack((dataset, np.array(lst, dtype=object)))
                    idx+=1
                   # else:
                    #    anomalies +=1
                     #   idx +=1

                    if idx % 100 ==0 :
                           print(idx, flush = True)

                except:
                    write = open(f'val_{i}.npy', 'wb')
                    np.save(write, dataset)
                    break

        #print('anomalies: ', anomalies)
        end = time.time()
        duration = end - start
        print(duration, 'second')
        #write = open('moreV.npy', 'wb')
        #np.save(write, dataset)
        print('Done', flush = True)


def extract_dataset_mp(path, name):
    print(name , flush = True)
    os.chdir(path)
    files = os.listdir()
    files = [f for f in files if f.endswith('.extxyz')]
    counter = 0
    dataset = np.array([], dtype=object)
    dataset = np.hstack((dataset, np.array(['mp_id= 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4', 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'task_id = 8'])))
 
    for i in files:
        name = i.replace('.extxyz', '')
        if counter % 1000 == 0:
            print(counter)
        counter +=1
        File = open(i, 'r')
        idx = 0
        while True:
                try:
                    structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
                    symbol = str(structure[0].symbols)
                    atoms = np.array(structure[0].get_chemical_symbols())
                    atoms = np.array( [ptable.index(j) for j in atoms])
                    natom = len(atoms)
                    cell = np.array(structure[0].get_cell())
                    pos = np.array(structure[0].get_positions())
                    energy = structure[0].info['corrected_total_energy']
                    force = np.array(structure[0].get_forces(apply_constraint=False))
                    Id = structure[0].info['task_id']
                    lst = [structure[0].info['mp_id'], natom, atoms, cell, symbol, pos, energy, force, Id]
                    dataset = np.vstack((dataset, np.array(lst, dtype=object)))
                    idx+=1
                except:
                    break

    write = open(f'{name}.npy', 'wb')
    np.save(write, dataset)
    print(i, ' Done', flush = True)


def extract_wbm_test_set(_file):
    f = json.load(open(_file, 'r'))
    dataset = np.array([], dtype=object)
    dataset = np.hstack((dataset, np.array(['idx= 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4', 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))
    
    material_ids = f['material_id'] # a dict
    material_formula = f['formula_from_cse']# a dict 
    systems = f['computed_structure_entry'] # a dict
    for _id in material_ids.keys():
        Id = _id
        structure = IStructure.from_dict(m['computed_structure_entry']['256962']['structure'])


def jmol_xyz_to_npy(path, name):
    '''
    converts jmol xyz files of QM9 dataset to npy files readable by LATTE
    '''
    atomic_ref = {1 : -0.500273,
                  6 : -37.846772,
                  7 : -54.583861,
                  8 :-75.064579,
                  9 : -99.718730}
    rm_list = np.load('/leonardo_scratch/large/userexternal/mtaleblo/Dataset/QM9/training_data/rm.npy', allow_pickle = True)
    print(rm_list[:3], flush = True)
    print(name, flush = True)
    os.chdir(path)
    _files = [f for f in os.listdir() if f.endswith('.xyz')]
    dataset = np.array([], dtype=object)
    dataset = np.hstack((dataset, np.array(['index = 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4', 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))
    Hartree_to_eV = 27.2114
    for _file in _files:
        _id = _file.replace('dsgdb9nsd_', '').replace('.xyz', '')
        if _id in rm_list:
            print(_id, ' skipped', flush = True)
            continue
        else:
            f = open(_file, 'r')
            r = f.readlines()
            natoms = int(r[0].replace('\n', ''))
# axs[0].set_yscale('log')            properties = r[1].split('\t')
            properties.pop()
            properties[0:1] = properties[0].split(' ')
            Internal_energy_0K = float(properties[12])
            idx = properties[1]
            forces = np.zeros((natoms, 3))
            Id = properties[0]
            atomic_numbers = []
            atomization_energy = Internal_energy_0K - (sum([atomic_ref[n] for n in atomic_numbers]))
            symbol = ''
            atomic_positions = []
            for i in range(2, natoms+2):
                line = r[i].replace('\n', '').split('\t')
                atomic_numbers.append(ptable.index(line[0]))
                cell = np.array([[5.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, 5.0]])
                symbol += line[0]
                atomic_positions.append([float(n.replace(' ','').replace('*^','e')) for n in line[1:4]])
                lst = [idx, natoms, np.array(atomic_numbers), cell, symbol, np.array(atomic_positions), atomization_energy * Hartree_to_eV , forces, Id]
            dataset = np.vstack((dataset, np.array(lst, dtype=object)))
    write = open(f'{name}_AE.npy', 'wb')
    np.save(write, dataset)
    print('Done', flush = True)


def wbm_to_npy(path):
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    from ase.io import extxyz    
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.atoms import Atoms
    from pymatgen.core.structure import IStructure

def Drautz_dataset(prefix, _path, _file, counter, idx0, idx):
    dataset = np.array([], dtype=object)
    dataset = np.hstack((dataset, np.array(['idx= 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4', 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))
    print('start', flush = True)
    for j in range(1,6):
        f = json.load(open(f'step_{j}.json', 'r'))
        f = f["entries"]
        for i in range(len(f)):
            material_id = 'wbm-' + str(j) + '-' + str(i)
            composition_dict = f[i]['composition'] # a dict
            system = f[i]['structure'] # a dict
            energy = f[i]['energy']
            natoms = int(sum(composition_dict.values()))
            atomic_numbers = []
            for k in composition_dict.keys():
                atomic_numbers.extend([ptable.index(k)]* int(composition_dict[k]))
            structure = AseAtomsAdaptor.get_atoms(IStructure.from_dict(system))
            pos = np.array(structure.get_positions())
            cell = np.array(structure.get_cell())
            forces = np.zeros((natoms, 3))
            symbol = str(structure.symbols)
            lst = [i, natoms, atomic_numbers, cell, symbol, pos,energy, forces, material_id]
            dataset = np.vstack((dataset, np.array(lst, dtype=object)))
            if i%5000==0:
                print(i, flush = True)

    write = open('wbm.npy', 'wb')
    np.save(write, dataset)
    print('Done', flush = True)


class Read_traj:

  def __init__(self, _file, step = -1):
    self.file = _file
    self.traj = T(_file)
    self.step = step

  def traj_length(self):
      return len(self.traj)

  def name(self):
    name =  self.file.replace('opt_', '').replace('.traj', '')
    print(name)
    return name

  def natoms(self):
    natoms = len(self.position())
    return natoms

  def energy(self): # eV
    energy = self.traj[self.step].get_total_energy()
  #   energy  = ! grep '!' $self.file | tail -1
  #   energy = float(energy[0].split('=')[-1].replace(' Ry', ''))
    return energy

  def force(self): # eV/ A
    force = self.traj[self.step].get_forces()
  #   d = ! grep -A 6 'Forces acting on atoms ' $self.file | tail -$(($self.natoms + 2))
  #   d = d[1:-1]
  #   force = np.array([[float(num) for num in  i.split('=')[-1].split(' ') if num != ''] for i in d])
    return force

  def position(self): # A
    position = self.traj[self.step].get_positions()
    return position

  def atomic_numbers(self):
    atomic_numbers = self.traj[self.step].get_atomic_numbers()
    return atomic_numbers

  def cell(self): # A
    cell = np.array(self.traj[self.step].get_cell())
    return cell

  def Chemical_composition(self):
    Chemical_composition = self.traj[self.step].symbols
    return Chemical_composition


################################## from traj to json
class traj_to_json:
  def __init__(self, pos, lattice, atomic_numbers, natoms, E, forces, key, pos_unit = 'cartesian', length_unit= 'angstrom'):
    self.pos = pos.tolist()
    self.lattice = lattice.tolist()
    self.atomic_numbers = atomic_numbers.tolist()
    self.natoms = natoms
    self.E = E
    self.forces = forces.tolist()
    self.key = key
    self.pos_unit = pos_unit
    self.length_unit = length_unit

  def build_atoms(self):
    atoms = []
    labels = list(range(self.natoms))
    system_atomic_symbols = [ptable[int(atom)] for atom in self.atomic_numbers]
    for idx, (atom, pos, force) in enumerate(zip(system_atomic_symbols, self.pos, self.forces)):
      species = atom
      atoms.append([labels[idx],species, pos, force])

    return atoms

  def dump_json(self):
    json_dict = {}
    json_dict["key"] = self.key
    json_dict["atomic_position_unit"] = self.pos_unit
    json_dict["unit_of_length"] = self.length_unit
    json_dict["energy"] = [self.E, "eV"]
    json_dict["lattice_vectors"] = self.lattice
    json_dict["atoms"] = self.build_atoms()

    with open(f'{self.key}.example', 'w') as json_file:
         json.dump(json_dict, json_file)
         print('Done')


# def read_pwo():
#      energy  = ! grep '!' $self.file | tail -1
#      energy = float(energy[0].split('=')[-1].replace(' Ry', ''))
#      d = ! grep -A 6 'Forces acting on atoms ' $self.file | tail -$(($self.natoms + 2))
#      d = d[1:-1]
#      force = np.array([[float(num) for num in  i.split('=')[-1].split(' ') if num != ''] for i in d])
# 


def Carbon_dataset_to_npy(prefix, _path, _file, counter, idx0, idx):
    print(f'Generating file {prefix}_{counter} from index {idx0} to {idx}.')
    File = open(_path + _file, 'r')
    dataset = np.array([], dtype=object)
    dataset = np.hstack((dataset, np.array(['category_name = 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4',
                 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))
 
    for idx in range(idx0,idx):
              structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
              category = structure[0].info['category']
              name = structure[0].info['name']
              symbol = str(structure[0].symbols)
              atoms = np.array(structure[0].numbers)
              natom = len(atoms)
              cell = np.array(structure[0].get_cell())
              pos = np.array(structure[0].get_positions())
              energy = structure[0].get_total_energy()
              force = np.array(structure[0].get_forces(apply_constraint=False))
              Id = str(counter) + '_' + str(idx) 
              lst = [category + '_'+ name, natom, atoms, cell, symbol, pos, energy, force, Id]
              dataset = np.vstack((dataset, np.array(lst, dtype=object)))
    write = open(_path + f'{prefix}_{counter}.npy', 'wb')
    np.save(write, dataset)
    print('Done', flush = True)

def Carbon_dataset_to_npy_categorised(prefix, _path, _file, cat_list, idx0, idx):
    os.chdir(_path)
    print(f'Going to dir: ', _path, flush = True)
    for cat in cat_list:
        print(cat,flush = True)
        c = 0
        print(f'Generating file {prefix}_{cat} from index {idx0} to {idx}.')
        File = open(_file, 'r')
        dataset = np.array([], dtype=object)
        dataset = np.hstack((dataset, np.array(['category_name = 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4',
                     'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))
     
        for idx in range(idx0,idx):
                  structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
                  category = structure[0].info['category']
                  if category == cat:
                      c +=1
                      if c% 500==0:
                          print(c, flush = True)
                      name = structure[0].info['name']
                      symbol = str(structure[0].symbols)
                      atoms = np.array(structure[0].numbers)
                      natom = len(atoms)
                      cell = np.array(structure[0].get_cell())
                      pos = np.array(structure[0].get_positions())
                      energy = structure[0].get_total_energy()
                      force = np.array(structure[0].get_forces(apply_constraint=False))
                      Id = cat + '_' + str(idx) 
                      lst = [category + '_'+ name, natom, atoms, cell, symbol, pos, energy, force, Id]
                      dataset = np.vstack((dataset, np.array(lst, dtype=object)))
      
        if '/' in cat:
            cat = cat.replace('/', '_')
            print(cat, flush = True)
        if ' ' in cat:
            cat = cat.replace(' ' , '_')
            print(cat, flush = True)

        write = open(f'{prefix}_{cat}.npy', 'wb')
        np.save(write, dataset)
        print('Done', flush = True)

def Drautz_dataset(prefix, _path, _file, counter, idx0, idx):
    print(f'Generating file {prefix}_{counter} from index {idx0} to {idx}.', flush = True)
    File = open(_path + _file, 'r')
    dataset = np.array([], dtype=object)
    dataset = np.hstack((dataset, np.array(['category_name = 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4',
                 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))

    for idx in range(idx0,idx):
              structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
              cat= np.array(structure[0].numbers)
              category = structure[0].info['category']
              name = structure[0].info['name']
              symbol = str(structure[0].symbols)
              atoms = np.array(structure[0].numbers)
              natom = len(atoms)
              cell = np.array(structure[0].get_cell())
              pos = np.array(structure[0].get_positions())
              energy = structure[0].info['REF_energy']
              force = structure[0].arrays['REF_forces']
              Id = str(counter) + '_' + str(idx)
              lst = [category + '_' + idx, natom, atoms, cell, symbol, pos, energy, force, Id]
#              if energy/natom < -0.5 :######## for filtered dataset
              dataset = np.vstack((dataset, np.array(lst, dtype=object)))
#              else:
#                  print(lst, flush= True)
    write = open(_path + f'{prefix}_{counter}.npy', 'wb')
    p.save(write, dataset)
    print('Done', flush = True)


############################################################# ref atomic energy
import torch
import torch.nn as nn

class LR(nn.Module):
        def __init__(self, n):
                super().__init__()
                self.linear = nn.Linear(n, 1) # One in and one out

        def forward(self, x):
                y_pred = self.linear(x)
                return y_pred


def l_reg(x, y, n, lr = 0.02):
    model = LR(n)
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    print('start')
    #x = x.view(-1, 1).float()  # Reshape x to be [batch_size, 1] and cast to float
    #y = y.view(-1, 1).float()

    old_loss = 0
    new_loss = 2000
    epoch = 0
    thr = 0.1
    while abs(new_loss - old_loss) > thr:
        old_loss = new_loss
        pred_y = model(x)
        new_loss = criterion(pred_y, y)
        optimizer.zero_grad()
        new_loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f'epoch {epoch}, loss: {new_loss}')
        epoch+=1
    print('epoch {}, loss {}'.format(epoch, new_loss.item()))
    print('parameters: ', model.state_dict())

###############################################################
def High_force_sys(forces, thr=11):
    mx = np.max(forces)
    if mx > thr:
        return True
    else:
        return False



import os
import ase
from ase.io import extxyz
import numpy as np
import time
import pickle
import json
from ase import io

class xyz_to_example:
    def __init__(self, _path, _file, f_filter = False):
        if _path.endswith('/'):
            self.path = _path
        else:
            self.path = _path + '/'
        self.file = _file
        self.File = self.load_xyz()
        self.pos_unit = 'cartesian'
        self.length_unit= 'angstrom'
        self.f_filter = f_filter

    def load_xyz(self):
        File = open(self.path + self.file, 'r')
        print(f'file {self.file}', flush = True)
        return File

    
    def read_system(self, idx):
        File = self.File
        structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
        symbol = str(structure[0].symbols)
        atoms = np.array(structure[0].get_chemical_symbols())
        atomic_numbers = np.array([ptable.index(j) for j in atoms])
        natoms = len(atomic_numbers)
        cell = np.array(structure[0].get_cell())
        pos = np.array(structure[0].get_positions())
        energy = get_energy(structure[0])
        force = get_forces(structure[0])
        try:
            category = structure[0].info['category']
        except KeyError: 
            category = 'unknown'
        #if (energy/natoms) > 0:
        #    continue
        if self.f_filter:
            if High_force_sys(force):
                pass
            else:
                return  [pos, cell, atomic_numbers, natoms, energy, force, category]

        else:
            return  [pos, cell, atomic_numbers, natoms, energy, force, category]

    

    
    def build_atoms(self, data, idx):
        pos = data[0].tolist()
        atomic_numbers = data[2].tolist()
        natoms = data[3]
        force = data[5].tolist()
    
        atoms = []
        labels = list(range(natoms))
        system_atomic_symbols = [ptable[int(atom)] for atom in atomic_numbers]
        for idx, (atom, pos, force) in enumerate(zip(system_atomic_symbols, pos, force)):
                species = atom
                atoms.append([labels[idx],species, pos, force])
    
        return atoms

    def dump_json(self, idx, data):           
        json_dict = {}
        json_dict["key"] = idx
        json_dict["atomic_position_unit"] = self.pos_unit
        json_dict["unit_of_length"] = self.length_unit
        json_dict["energy"] = [data[4], "eV"]
        json_dict["lattice_vectors"] = data[1].tolist()
        json_dict["atoms"] = self.build_atoms(data, idx)
        id = self.file.split('.')[0] 
        with open(f'{data[-1].replace("/", "_").replace(" ", "")}_{idx}.example', 'w') as json_file:
            json.dump(json_dict, json_file)
            if idx % 100 == 0 :
               print(f'{id}_{idx}.example  Done', flush = True)
        

    def count_atomic_structures_ase(self):
        structures = io.read(self.path + self.file, index=':')
        return len(structures)

    def ref_atomic_energy(self, ref_energy):
        from collections import Counter
        atoms = []
        energies = []
        elements = {}
        for sys in ref_energy:
            atoms.append([v for v in Counter(sys[0]).values()])
            energies.append(sys[1])
            for i in sys[0]:
                elements[ptable[i]]=None
        print('elements are: ', elements.keys(), flush = True)
        n_atoms = []
        n_elements = len(elements.keys())
        for atm in atoms:
            if len(atm) < n_elements:
                atm.extend([0] * (n_elements - len(atm)))
            n_atoms.append(atm)
        atoms_tensor = torch.tensor(n_atoms, dtype=torch.float32)
        energies_tensor = torch.tensor(energies, dtype=torch.float32).reshape(-1, 1)
        l_reg(atoms_tensor, energies_tensor, n_elements)


    def work_flow(self):
        all_idx = self.count_atomic_structures_ase()
        ref_energy = []
        if not os.path.exists(self.path + f'{self.file.split(".")[0]}_examples'):
            os.makedirs(self.path + f'{self.file.split(".")[0]}_examples')
            
        os.chdir(self.path + f'{self.file.split(".")[0]}_examples')
 
        for idx in range(2300, all_idx):
            data = self.read_system(idx)
            # remove systems containing N
            #if 7 in data[2]:
            #    continue
            if data:
                self.dump_json(idx, data)
                ref_energy.append([data[2], data[4]])
        os.chdir(self.path)
        self.ref_atomic_energy(ref_energy)

        if self.File:
            self.File.close()
            os.chdir(self.path)
        


