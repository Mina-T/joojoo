from ase.io import extxyz
import numpy as np
from M_dataset import *

#p='/leonardo_scratch/large/userexternal/mtaleblo/Carbon_project/Dataset/Simone/Drautz/spin/'
#_file = 'C_Database_IOM_Spin.xyz'
#n_structures = 14790 #spin

p='/leonardo_scratch/large/userexternal/mtaleblo/Carbon_project/Dataset/Drautz/'
_file = 'test_all.xyz'
n_structures = 1721

indexes = [i for i in range(0, n_structures+1, test_size)]
counters = [i for i in range(len(indexes)-1)]
prefix = 'Drautz_test_again'
for c in counters:
    Drautz_dataset(prefix, p , _file, c, indexes[c], indexes[c+1])

#n_structures = 17141  # non_spin
#n_structures = 14790 #spin 
#n_structures = 15464 Drautz train
#test_size = int(n_structures * 0.1)


#for c in counters:
#    Carbon_dataset_to_npy(prefix, Simone_p + 'Drautz/spin/' , _file, c, indexes[c], indexes[c+1])


# unbiased, categorized
#prefix = 'IOM_Spin'
#for i in range(n_structures): 
#    Carbon_dataset_to_npy_categorised(prefix, p , _file, ['amorphous/liquid', 'general bulk', 'general clusters', 'sp2 bonded', 'sp3 bonded'], 0, n_structures)
