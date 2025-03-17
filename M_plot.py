from random import randint  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from M_validation_analysis import *
from M_dataframe import read_metrics, read_dat
from M_models import *

def plot_df(df, axs=None ,label=None, color=None, factor = 1, validation = True, train = True):
    try:
        if not color:
            color = ('#%06X' % randint(0, 0xFFFFFF))
        if validation == True:
            try:
                axs[0].plot(df['#epoch'] * factor, df['VMAE/at'], color= color, alpha = 0.7 , ms = 6.0, label = label)
                axs[1].plot(df['#epoch'] * factor, df['VMAEF'], color= color, alpha = 0.7, ms = 6.0) #+  ' V'
            except:
                axs[0].plot(df['#epoch'] * factor, df['Val_MAE/at'], color= color, alpha = 0.7 , ms = 6.0, label = label)
                axs[1].plot(df['#epoch'] * factor, df['Val_MAEF'], color= color, alpha = 0.7, ms = 6.0)
        if train == True:
                axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.4, ms = 0.08)
                axs[1].plot(df['#epoch']* factor, df['MAEF'], color= color , alpha = 0.4, ms = 0.08, label = label)

    except:
        if not color:
            color = ('#%06X' % randint(0, 0xFFFFFF))
        if validation == True:
            axs[0].plot(df['#epoch'] * factor, df['Val_MAE/at'], color= color, alpha = 0.7 , ms = 6.0, label = label)
            axs[1].plot(df['#epoch'] * factor, df['Val_MAEF'], color= color, alpha = 0.7, ms = 6.0) #+  ' V'
        if train == True:
            axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.4, ms = 0.08)
            axs[1].plot(df['#epoch']* factor, df['MAEF'], color= color , alpha = 0.4, ms = 0.08, label = label)
#    print('min E_err: ', df['Val_MAE/at'].min(), 'at epoch:', df['Val_MAE/at'].idxmin(), flush = True) 
#    print('min F_err: ', df['Val_MAEF'].min(), 'at epoch:', df['Val_MAEF'].idxmin(), flush = True)



def plot_E_train(df, axs=None ,label=None, color=None, factor = 1, validation = True, train = True):
    try:
        if not color:
            color = ('#%06X' % randint(0, 0xFFFFFF))
        if validation == True:
            try:
                axs[0].plot(df['#epoch'] * factor, df['VMAE/at'], color= color, alpha = 0.8 , ms = 6.0,  label = label)
            except:
                axs[0].plot(df['#epoch'] * factor, df['Val_MAE/at'], color= color, alpha = 0.8 , ms = 6.0,  label = label)
        if train == True:
                axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.5, ms = 0.08,  label = label)
    except:
        if not color:
            color = ('#%06X' % randint(0, 0xFFFFFF))
        if validation == True:
            axs[0].plot(df['#epoch'] * factor, df['Val_MAE/at'], color= color, alpha = 0.8 , ms = 6.0,  label = label)
        if train == True:
            axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.5, ms = 0.08,  label = label)


def plot_validation(path, epoch, label=None,title= None, shift = 0, x0 = None, x = None, f_lim = False):
    fig, axs = plt.subplots(1, 2, figsize = (12, 5))
    e_file = read_dat(path, f'epoch_{epoch}_step_{1000*epoch}.dat')
    f_file = read_dat(path, f'epoch_{epoch}_step_{1000*epoch}_forces.dat')
    axs[0].plot(e_file['e_ref']/e_file['n_atoms'], (e_file['e_nn']/e_file['n_atoms']), '.r', label = label)
    e_atm = e_file['e_nn']/e_file['n_atoms']
    e_atm_ref = e_file['e_ref']/e_file['n_atoms']
    axs[0].plot([min(e_atm_ref), max(e_atm_ref)], [min(e_atm), max(e_atm)], color = 'k')
    axs[0].set_title('Energy (eV/atom)')

    axs[1].plot(f_file['fx_ref'], f_file['fx_nn'], '.r', label = label)
    axs[1].plot([min(f_file['fx_ref']), max(f_file['fx_ref'])], [min(f_file['fx_ref']), max(f_file['fx_ref'])], color = 'k')
    axs[1].set_title('Force/Atom (eV/A)')
    axs[0].set_xlim(x0,x)
    axs[0].set_ylim(x0,x)
    if f_lim:
        axs[1].set_xlim(x0,x)
        axs[1].set_ylim(x0,x)
    e_mae = E_MAE(e_file) + shift
    f_mae = F_MAE(f_file)
    e_rmse = RMSE_E(e_file)
    f_rmse = RMSE_F(f_file)
    axs[0].annotate(f'MAE = {e_mae:.3f}', xy=(0.2,0.8), xycoords='figure fraction')
    axs[1].annotate(f'MAE = {f_mae:.3f}', xy=(0.6,0.8), xycoords='figure fraction')
    axs[0].annotate(f'RMSE = {e_rmse:.3f}', xy=(0.2,0.7), xycoords='figure fraction')
    axs[1].annotate(f'RMSE = {f_rmse:.3f}', xy=(0.6,0.7), xycoords='figure fraction')
    #fig.tight_layout(pad=5.5)
    fig.supxlabel('reference', size = 16)
    fig.supylabel('LATTE', size = 16)
    if title:
        plt.suptitle(f'{title}', fontsize = 18)
    plt.legend()
    plt.show()


def plot_setup(fig, axs, log = True, lim = True, x0lim = 10,  xlim = 1000, y0lim = 0.05, ylim = 0.3, fylim =0.55 ):
    axs[0].legend(loc = 'upper right', prop={'size': 12})
    axs[1].legend(loc = 'upper right', prop={'size': 12})
    for ax in [axs[0], axs[1]]:
        leg = ax.legend()
        for line in leg.get_lines():
            line.set_linewidth(6.0)
    
    axs[0].set_title('Energy', fontsize= 18)
    axs[1].set_title('Force', fontsize = 18)
    axs[0].set_ylabel('MAE (eV)')
    axs[1].set_ylabel('MAE (eV/A)')
    fig.supxlabel('log Epochs(1000 steps each)')
    if lim:
        axs[0].set_xlim(x0lim,xlim)
        axs[0].set_ylim(y0lim, ylim)
        axs[1].set_xlim(x0lim,xlim)
        axs[1].set_ylim(y0lim, fylim)

    minor_ticks = np.arange(0, 0.1, 0.005)
    axs[0].set_yticks(minor_ticks, minor=True)
    axs[0].grid(which='minor', axis = 'y', alpha=0.2)
    axs[0].grid(which='major', axis = 'y', alpha=0.7)

    minor_ticks = np.arange(0.05, 0.2, 0.005)
    axs[1].set_yticks(minor_ticks, minor=True)
    axs[1].grid(which='minor', axis = 'y', alpha=0.2)
    axs[1].grid(which='major', axis = 'y', alpha=0.7)
    if log:
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
    plt.show()


def plot_learning_curve(labels:list, h_t, h_in, h_out, emb_t, emb_in, emb_out):
    x = [i for i in range(1, len(labels)+1)]
    plt.plot(x, emb_t, 'r', label ='Training')
    plt.plot(x, emb_in, 'c', label ='In domain')
    plt.plot(x, emb_out, 'g', label = 'Out of domain')

    plt.plot(x, h_t, '-r',linestyle='dashed')
    plt.plot(x, h_in, '-c', linestyle='dashed')
    plt.plot(x, h_out, '-g', linestyle='dashed')

    plt.ylim(-0.005, 0.13)
    plt.xticks(ticks = x, labels = labels)
    plt.xlabel('Training set size', fontsize = 12)
    plt.ylabel('Error (meV/atm)', fontsize = 12)
    plt.legend()
    plt.show()


def plot_bar_cat_err(categories:dict):
    '''
    Generate a bar plot for {category:{category_train_n, category_test_n, category_error}}
    '''
    pass


def plot_valiated_model(axs, _path, color = 'r', label = None):
     E_mae, F_mae = [], []
     p_v = _path
     epochs = get_sort_models(p_v , 'dat')
     for epoch in epochs:
         E_mae.append(E_MAE(read_dat(p_v,f'epoch_{epoch}_step_{epoch*1000}.dat' )))
         F_mae.append(F_MAE(read_dat(p_v,f'epoch_{epoch}_step_{epoch*1000}_forces.dat')))
     axs[0].scatter(epochs, E_mae, color = color)
     axs[1].scatter(epochs, F_mae, color = color, label = label)


import glob
def plot_training_err_sevenNet(_path, label = None, y0lim = 5, y1lim = 0.4, y2lim = 0.4):
    files = glob.glob(f'{_path}/*.csv')
    all_df = []
    for f in files:
        df = pd.read_csv(f)
        all_df.append(df)
    all_df = pd.concat(all_df, ignore_index=False)
    all_df = all_df.sort_values(by='epoch', ascending=True)
    all_df = all_df.reset_index(drop=True)

    fig, axs = plt.subplots(1, 3, figsize = (15, 5))
    axs[0].plot(all_df['epoch'], all_df['trainset_TotalLoss'], 'r', alpha = 0.3)
    axs[0].plot(all_df['epoch'], all_df['validset_TotalLoss'], 'r', label='Loss')

    axs[1].plot(all_df['epoch'], all_df['trainset_Energy_MAE'], 'g', alpha = 0.3)
    axs[1].plot(all_df['epoch'], all_df['validset_Energy_MAE'], 'g', label='energy MAE (eV)')

    axs[2].plot(all_df['epoch'], all_df['trainset_Force_MAE'], 'b', alpha = 0.3)
    axs[2].plot(all_df['epoch'], all_df['validset_Force_MAE'], 'b', label='forces MAE eV/A')

    axs[0].set_ylabel('Loss')
    axs[1].set_ylabel('MAE (meV)')
    axs[2].set_ylabel('MAE (meV /A)')
    axs[0].set_ylim(0, y0lim)
    axs[1].set_ylim(0, y1lim)
    axs[2].set_ylim(0, y2lim)

    axs[1].set_xlabel('epoch')
    for ax in axs.flatten():
        ax.legend()
    plt.suptitle(label)
    plt.show()

