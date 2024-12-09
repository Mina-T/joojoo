from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from M_dataframe import read_metrics, read_dat
import json
import glob
import os

def plot_err_MACE(_path, label = None, y0lim = 1, y1lim = 0.1, y2lim = 0.15):
    loss, mae_e_per_atom, mae_f, epochs = [], [], [], []
    _file = glob.glob(os.path.join(_path, "*_train.txt"))
    with open(_file[0], 'rb') as f:
        for line in f:
            c = 0
            read_line = json.loads(line)
            if "mae_e_per_atom" in read_line.keys():
                if read_line['epoch'] =="null" and c >=1:
                    pass
                elif read_line['epoch'] =="null" and c < 1:
                    loss.append(read_line['loss'])
                    mae_e_per_atom.append(read_line['mae_e_per_atom'])
                    mae_f.append(read_line['mae_f'])
                    epochs.append(read_line['epoch'])
                
                else:
                    loss.append(read_line['loss'])
                    mae_e_per_atom.append(read_line['mae_e_per_atom'])
                    mae_f.append(read_line['mae_f'])
                    epochs.append(read_line['epoch'])
                            
    fig, axs = plt.subplots(1, 3, figsize = (15, 5))
    axs[0].plot(epochs, loss, 'r', label='Loss')
    axs[1].plot(epochs, mae_e_per_atom, 'g', label='mae_atom')
    axs[2].plot(epochs, mae_f, 'b', label='mae_f')
    axs[0].set_ylabel('Loss')
    axs[1].set_ylabel('MAE (meV)')
    axs[2].set_ylabel('MAE (meV /A)')
    axs[0].set_ylim(0, y0lim)
    axs[1].set_ylim(0, y1lim)
    axs[2].set_ylim(0.05, y2lim)

    axs[1].set_xlabel('epoch')
    for ax in axs.flatten():
        ax.legend()
    plt.suptitle(label)
    plt.show()



