import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt


def read_metrics(path):
    """
    Given a path, converts the metrics.dat file into a pandas DataFrame.
    """
    df = pd.read_csv(f"{path}/metrics.dat", delim_whitespace=True, usecols=['#epoch'])
    df = df.groupby('#epoch', as_index=False).last()
    return df



def read_dat(path, file):
    '''
    Given a path, converts the dat into df
    '''
    _file = os.path.join(path, file)
    df= pd.read_csv(_file, delim_whitespace = True )
    return df
