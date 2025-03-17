import os
import pickle
import numpy as np
import torch
from ase.io import read, write
import pandas as pd
import matplotlib.pyplot as plt

def MACE_pt_to_model(_path, save_path):
    files = os.listdir(_path)
    for f in files:
        if f.endswith('.pt'):
            ckpt = torch.load(os.path.join(_path, f), map_location=torch.device('cpu'))
            model = torch.load(os.path.join(_path, "MACE_model_run-123.model"), map_location=torch.device('cpu'))
            model.load_state_dict(ckpt["model"])
            name = f.replace('.pt', '.model')
            torch.save(model, os.path.join(p, save_path, name))



def validate_mace(f_ref, f_mace):
    '''
    Given the reference and validated files, returns a dataframe including ref_E and mace_E per atom.
    '''
    ref_data = read(f_ref, index=':')
    E_ref = {}
    for sys in ref_data:
        idx = ref_data.index(sys)
        cat = sys.info['category']
        n=len(sys.get_atomic_numbers())
        e = sys.get_total_energy()
        E_ref[f'{idx}_{cat}'] = e/n

    df = pd.DataFrame({'Mace': [None]*len(E_ref), 'Ref': [None]*len(E_ref)})
    E_ref_df = pd.DataFrame(list(E_ref.items()), columns=['idx', 'Ref'])
    E_ref_df.set_index('idx', inplace=True)

    mace_data = read(f_mace, index=':')
    E_mace = {}
    for sys in mace_data:
        idx = ref_data.index(sys)
        cat = sys.info['category']
        n=len(sys.get_atomic_numbers())
        e = sys.info['MACE_energy']
        E_mace[f'{idx}_{cat}'] = e/n
    E_mace_df = pd.DataFrame(list(E_mace.items()), columns=['idx', 'Mace'])
    E_mace_df.set_index('idx', inplace=True)
    df = pd.merge(E_ref_df, E_mace_df, left_index=True, right_index=True, how='outer')
    return df


def get_high_E_err_systems(_dir, f_ref, err_threshold):
    '''
    returns a pd.df of systems which their error is higher than a threshold during training
    '''
    high_err_df = pd.DataFrame([])
    files = os.listdir(_dir)
    files = [f for f in files if f.endswith('.model.xyz')]

    epochs = sorted([int(f.replace('output_', '').replace('_swa.model.xyz', '')) for f in files])
    epochs = epochs[::2]
    for epoch in epochs:
        df = validate_mace(f_ref, os.path.join(_dir, f'output_{epoch}_swa.model.xyz'))
        err = df[abs((df['Ref'] - df['Mace']))> err_threshold]
        if epoch == epochs[0]:
            high_err_df[f'error_{epoch}'] = df.index.str
        high_err_df[f'error_{epoch}'] = abs((err['Ref'] - err['Mace']))

    columns = [f'error_{epoch}' for epoch in epochs]
    high_err_df_final = high_err_df.dropna(subset=columns)
    for c in columns[:]:
        plt.hist(high_err_df_final[c], label = c, histtype=u'step', density=True)
    plt.legend()
    plt.yscale('log')
    return high_err_df_final

