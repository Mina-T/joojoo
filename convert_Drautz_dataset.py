from ase.io import extxyz
import numpy as np
from M_dataset import Carbon_dataset_to_npy

Simone_p = '/leonardo_scratch/large/userexternal/mtaleblo/Carbon_project/Dataset/Simone/'
_file = 'C_Database_IOM_Spin.xyz'
#n_structures = 17141   non_spin
n_structures = 14790
test_size = int(n_structures * 0.1)

indexes = [i for i in range(0, n_structures+1, test_size)]
counters = [i for i in range(len(indexes)-1)]

prefix = 'IOM_spin'
for c in counters:
    Carbon_dataset_to_npy(prefix, Simone_p + 'Drautz/spin/' , _file, c, indexes[c], indexes[c+1])
