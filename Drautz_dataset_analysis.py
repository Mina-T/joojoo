from ase.io import read
from ase.io import extxyz
import numpy as np
import sys
from M_dataset import get_energy, get_forces

p = sys.argv[1]
_file = sys.argv[2]
#cat = sys.argv[3]
categories = {}
EF = []

File = open(p + '/' + _file, 'r')
n_structures = len(read(File, index=':'))
print('Reading from ', p , 'file: ', _file, flush = True)
idx = 0
for i in range(n_structures):
    structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
    idx +=1
    category = structure[0].info['category']
    #if cat in category:
        #print(idx, flush = True)
    atoms = structure[0].get_atomic_numbers()
    energies =  get_energy(structure[0]) # for Drautz dataset
    forces = get_forces(structure[0])# for Drautz dataset
    EF.append([energies, forces, atoms,  category, idx])

EF = np.array(EF, dtype = object)
np.save(open('All.npy', 'wb'), EF)
print('Done', flush = True)



def get_drautz_label(key):
    for label in ['cluster', 'sp2', 'sp3', 'bulk', 'amorph']:
        if label in key:
            return label
    return None
