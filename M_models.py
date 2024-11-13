import os
import glob
import shutil

def get_sort_models(_dir, format='pkl'):
    """
    Given a directory and a file format, return a sorted list of epoch numbers 
    based on filenames like "model_epoch_0.pkl".
    """
    files = glob.glob(os.path.join(_dir, f"*0.{format}"))
    epochs = sorted([int(f.split('_')[1]) for f in files])
    
    return epochs


def remove_models(_dir, freq):
    '''
    Get sorted epochs, keep models at every freq as well as last model, remove the rest. 
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
    '''
    called inside models/keep directory to move selected models from /keep to /models directory.
    '''
    epochs = get_sort_models(_dir)
    shutil.move(f'./epoch_{epochs[-1]}_step_{epochs[-1]*1000}.pkl', f'../epoch_{epochs[-1]}_step_{epochs[-1]*1000}.pkl')
    for e in epochs[::freq]:
        shutil.move(f'./epoch_{e}_step_{e*1000}.pkl', f'../epoch_{e}_step_{e*1000}.pkl')
        print(f'epoch_{e}_step_{e*1000}.pkl  moved.')

