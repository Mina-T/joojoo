import os
import pickle
import numpy as np
import torch
from ase.io import read, write
import pandas as pd
import matplotlib.pyplot as plt
import re

def MACE_pt_to_model(_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files = os.listdir(_path)
    files = [f for f in files if f.endswith('.pt')]
    if len(files) > 15:
        step  = len(files) // 15
        files = files[::step+1]
    for f in files:
            ckpt = torch.load(os.path.join(_path, f), map_location=torch.device('cpu'))
            model = torch.load(os.path.join(_path, "MACE_model_run-123.model"), map_location=torch.device('cpu'))
            model.load_state_dict(ckpt["model"])
            name = f.replace('.pt', '.model')
            torch.save(model, os.path.join(_path, save_path, name))



def validate_mace(file_ref, file_mace):
    '''
    Given the reference and validated files, returns a dataframe including ref_F and mace_F.
    '''
    ref_data = read(file_ref, index=':')
    E_ref = {}
    F_ref = {}
    for sys in ref_data:
        idx = ref_data.index(sys)
        cat = sys.info['category']
        n=len(sys.get_atomic_numbers())
        e = sys.get_total_energy()
        force = sys.get_forces()
        fx = force[:,0]
        E_ref[f'{idx}_{cat}'] = e/n
        for atom_idx in range(len(fx)):
            F_ref[f'{idx}_{cat}_{atom_idx}'] = fx[atom_idx]
   
    mace_data = read(file_mace, index=':')
    E_mace = {}
    F_mace = {}
    for sys in mace_data:
        idx = ref_data.index(sys)
        cat = sys.info['category']
        n=len(sys.get_atomic_numbers())
        e = sys.info['MACE_energy']
        E_mace[f'{idx}_{cat}'] = e/n
        force = sys.arrays['MACE_forces']
        fx = force[:,0]
        for atom_idx in range(len(fx)):
            F_mace[f'{idx}_{cat}_{atom_idx}'] = fx[atom_idx]

    E_df = pd.DataFrame([E_ref, E_mace]).T.rename(columns={0: 'E_ref', 1: 'E_mace'})
    F_df = pd.DataFrame([F_ref, F_mace]).T.rename(columns={0: 'F_ref', 1: 'F_mace'})
    return E_df, F_df
 

def get_err_during_training(_dir, file_ref):
    '''
    returns a pd.df of systems whith their error during training.
    _dir: directory containing validation files
    file_ref: DFT ref file (dataset) 
    '''
    E_err_df = pd.DataFrame()
    F_err_df = pd.DataFrame()
    files = os.listdir(_dir)
    files = [f for f in files if f.endswith('.xyz') and re.search(r'\d', f)]

    epochs = sorted([int(re.findall(r'\d+', f)[0]) for f in files])
    epochs = epochs[::4]
    for epoch in epochs:
        print(f'validating {epoch}', flush= True)
        E_df, F_df = validate_mace(file_ref, os.path.join(_dir, f'output_{epoch}.xyz'))
        E_err = abs(E_df['E_ref'] - E_df['E_mace'])
        F_err = abs(F_df['F_ref'] - F_df['F_mace'])
        if epoch == epochs[0]:
            E_err_df['file_name'] = E_df.index.astype(str)
            F_err_df['file_name'] = F_df.index.astype(str)
        E_err_df[f'error_{epoch}'] = E_err.values
        F_err_df[f'error_{epoch}'] = F_err.values
    E_err_df.set_index('file_name', inplace=True)
    F_err_df.set_index('file_name', inplace=True)
    E_err_df_final = E_err_df.dropna()
    F_err_df_final = F_err_df.dropna()
    return E_err_df_final, F_err_df_final

