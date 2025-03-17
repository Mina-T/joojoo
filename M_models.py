import os
import glob
import shutil


def backup(_dir):
    '''
    1. keep some of the models in models directory.
    
    '''
    pass

def get_sort_models(_dir, format = 'pkl'):
    os.chdir(_dir)
    files = glob.glob(f"*0.{format}")
    epochs = []
    for f in files:
        epochs.append(int(f.split('_')[1]))
    epochs = sorted(epochs)
    return epochs
 

def remove_models(_dir, freq):
    '''
    Get sorted epochs, keep every freq and last model, remove the rest. 
    '''
    epochs = get_sort_models(_dir)
    epochs.pop()
    epochs.pop(-5)
    epochs.pop(1)
    for e in epochs[::freq]:
        epochs.remove(e)
    for e in epochs:
        os.remove(f'epoch_{e}_step_{e*1000}.pkl')
    print('Done')


def move_models(_dir, freq:int):
    files = glob.glob("*0.pkl")
    epochs = []
    for f in files:
        epochs.append(int(f.split('_')[1]))
    epochs = sorted(epochs)
    shutil.move(f'./epoch_{epochs[-1]}_step_{epochs[-1]*1000}.pkl', f'../epoch_{epochs[-1]}_step_{epochs[-1]*1000}.pkl')
    for e in epochs[::freq]:
        shutil.move(f'./epoch_{e}_step_{e*1000}.pkl', f'../epoch_{e}_step_{e*1000}.pkl')
        print(f'epoch_{e}_step_{e*1000}.pkl  moved.')

