import numpy as np
import pandas as pd
import os
import pickle
from M_dataframe import *
from random import randint
from M_validation_analysis import *


class Error_decomposition:
    def __init__(self, models, epoch):
        self.models = models
        self.epoch = epoch

    
    def load_data(self):
        '''
        For m models returns [{Id:E_nn} * m]   &   Avg={Id : avg error}
        '''
        data = {}
        model_counter = 1
        for model in self.models:
            p = model 
            df = pd.read_csv(p + f'epoch_{self.epoch}_step_{1000*self.epoch}.dat', delim_whitespace = True)
            data[model_counter] = {row['#filename']: {'e_nn': row['e_nn'] / row['n_atoms'],
                                                      'e_ref': row['e_ref'] / row['n_atoms']} for _, row in df.iterrows()}
        return data


    def avg_model(self):
        E_ref = {}
        E_of_models = []

        for model_data in self.data.values():
            E_nn = {}
            for Id, values in model_data.items():
                E_nn[Id] = values['e_nn']
                E_ref[Id] = values['e_ref']
            E_of_models.append(E_nn)

        Avg_model = {Id: np.mean([E_nn[Id] for E_nn in E_of_models]) for Id in E_ref.keys()}

        return E_ref, E_of_models, Avg_model


    
    
    def config_variance(self):
        '''
        Returns variance = {Id: var}
        '''
        print('epoch ', self.epoch)
        E_ref, E_of_models, Avg_model = self.avg_model()
        # Ids_nn = {}
        # for Id in E_ref.keys():
        #     Id_E_nn = [E_of_models[i][Id] for i in range(len(E_of_models))]
        #     Ids_nn[Id] = (Id_E_nn)
            
        Id_variance = {}
        for Id in E_ref.keys():
            Id_avg = Avg_model[Id]
            Id_errs = [E_of_models[i][Id] for i in range(len(E_of_models))]
            diff = [(Id_err - Id_avg)**2 for Id_err in Id_errs]
            variance = np.mean(diff)
            Id_variance[Id] = variance
        avg_var = np.mean(Id_variance.values())
        # print(f'Average variance of the models at epoch {self.epoch} is {avg_var} eV')   
        return Id_variance, avg_var

    def config_bias(self):
        E_ref, E_of_models, Avg_model = self.avg_model()
        Id_bias = {}
        for Id in E_ref.keys():
            Id_avg = Avg_model[Id]
            Id_ref = E_ref[Id]
            diff = (Id_avg - Id_ref)**2
            Id_bias[Id] = diff
            
        avg_bias = np.mean(Id_bias.values())
        # print(f'Average bias of the models at epoch {self.epoch} is {avg_bias} eV')   
        return Id_bias, avg_bias
        
            
