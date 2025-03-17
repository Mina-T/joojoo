from ase.io import extxyz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import math
from M_validation_analysis import *
from M_dataframe import *
from M_models import *

def get_high_err_systems(_dir, epochs, err_threshold, ignore_initial_epochs= 2):
    '''
    returns a list of systems id which their error is higher than a threshold during training
    '''
    high_err_df = pd.DataFrame([])
    for epoch in epochs:
        df = read_dat(_dir, f'epoch_{epoch}_step_{epoch*1000}_forces.dat')
        err = df[abs(df['fx_ref'] - df['fx_nn'])> err_threshold]
        if epoch == epochs[0]:
            high_err_df[f'#filename_{epoch}'] = err['#filename']
        high_err_df[f'error_{epoch}'] = abs(err['fx_ref'] - err['fx_nn'])

    columns = [f'error_{epoch}' for epoch in epochs]
    high_err_df_final = high_err_df.dropna(subset=columns[ignore_initial_epochs:])
    return high_err_df_final
    #try:
    #    ids = high_err_df_final[f'#filename_{epochs[0]}'].to_numpy()
    #    ids = [_id.split('_', 1)[1:][0] for _id in ids]
    #    print('npy')
    #except:
    #    ids = high_err_df_final[f'#filename_{epochs[0]}'].to_numpy()
    #    print('example')
