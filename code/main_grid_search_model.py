"""
Created on July 10 2021

@author: Elia Lombardo

Main script to perform a hyper-parameter search for a given model
"""
# %%
# IMPORT MODULES

# self written modules & parameters from config
import config
if config.grid_search is False:
    raise Exception('Attention: grid search code is run!')
from config import gpu_usage
from config import device
from config import x_train_snippets, y_train_snippets
from config import x_val_snippets, y_val_snippets
from config import input_size, hidden_size, num_layers
from config import wdw_size_i, wdw_size_o
from config import lr, epochs, batch_size, l2, bi
from config import net, loss_fn, dropout, early_stopping
from config import phase, direction, breathhold_inclusion, curve_aspect
from config import normalization, cohort, train_split, val_split, test_split
from auxiliary import plotting, architectures, utils
from auxiliary import train_val_test, data_preparation

import time
import numpy as np
import os
import torch
from sklearn.model_selection import ParameterGrid

# set hyper-parameters over which grid search is performed
parameters = {'num_layers': num_layers, \
              'batch_size': batch_size, \
              'lr': lr, \
              'dropout': dropout, \
              'l2': l2, \
              'val_RMSE_poi': [0], }

# create list with combination of hyperparameters and corresponding entry for metric set to zero
list_params = list(ParameterGrid(parameters))
print(list_params)
print('\n')

# %%
# PREPARE FOLDER FOR RESULTS

# set strings for results        
other_params = f'epochs={epochs}-wdw_size_i={wdw_size_i}-wdw_size_o={wdw_size_o}' + \
                f'-step_size={1}-direction={direction}' + \
                f'-breathhold_inclusion={breathhold_inclusion}' + \
                f'\n -curve_aspect={curve_aspect}-normalization={normalization}-cohort={cohort}' + \
                f'\n -train_split={train_split}-val_split={val_split}-test_split={test_split}'

# create folder for results          
path_saving = os.path.join(config.path_results, net, phase, 
                           time.strftime('%Y-%m-%d-%H:%M:%S') + '_grid_search')          
os.makedirs(path_saving, exist_ok=True)

# keep track of starting time
t0 = time.time()
# %%
# GRID SEARCH

for combination in range(len(list_params)):
    print("==============================================")
    print("===== Current Combination => %d/%d ============" % (combination + 1, len(list_params)))
    print("==============================================")
    
    # get network hyper-params for current combination
    current_net_params = f'num_layers={list_params[combination]["num_layers"]}' + \
                f'-batch_size={list_params[combination]["batch_size"]}' + \
                f'-lr={list_params[combination]["lr"]}' + \
                f'-l2={list_params[combination]["l2"]}' + \
                f'-dropout={list_params[combination]["dropout"]}'
                
    # create subfolder for results for current net parameters
    path_saving_subfolder = os.path.join(path_saving, current_net_params)
    os.makedirs(path_saving_subfolder, exist_ok=True)
        
        
    # PREPARE DATA FOR OPTIMIZATION
    # get batches of data for current combination
    x_train_batches = data_preparation.get_data_batches(data=x_train_snippets, 
                                                        batch_size=list_params[combination]['batch_size'], 
                                                        concat=True,
                                                        to_tensor=True, 
                                                        gpu_usage=gpu_usage, device=device)
    y_train_batches = data_preparation.get_data_batches(data=y_train_snippets, 
                                                        batch_size=list_params[combination]['batch_size'],  
                                                        concat=True,
                                                        to_tensor=True,
                                                        gpu_usage=gpu_usage, device=device)
    print('\n')
    print(f'Shape of obtained x_train_batches: {x_train_batches.shape}')  # (nr_batches, batch_size, wdw_size_i)
    print(f'Shape of obtained y_train_batches: {y_train_batches.shape}')  # (nr_batches, batch_size, wdw_size_o)
    print('\n')

    x_val_batches = data_preparation.get_data_batches(data=x_val_snippets, 
                                                    batch_size=list_params[combination]['batch_size'],                                                       
                                                    concat=True,
                                                    to_tensor=True, 
                                                    gpu_usage=gpu_usage, device=device)
    y_val_batches = data_preparation.get_data_batches(data=y_val_snippets,
                                                    batch_size=list_params[combination]['batch_size'],                                                        
                                                    concat=True,
                                                    to_tensor=True, 
                                                    gpu_usage=gpu_usage, device=device)
        
    
    # MODEL BUILDING AND OPTIMIZATION
    if net == 'LSTM_stateless':
        # create LSTM instance
        model = architectures.CentroidPredictionLSTM(input_size=input_size, hidden_size=hidden_size, 
                                        num_layers=list_params[combination]['num_layers'], 
                                        batch_size=list_params[combination]['batch_size'], 
                                        seq_length_in=wdw_size_i, seq_length_out=wdw_size_o,
                                        dropout=list_params[combination]['dropout'], bi=bi,
                                        gpu_usage=gpu_usage, device=device)
    else:
        raise Exception('Attention: unknown net name!')


    if gpu_usage:
        model.to(device)
    print(model)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=list_params[combination]['lr'],
                                 weight_decay=list_params[combination]['l2'])

    # train and validate the model
    train_losses, val_losses, val_losses_poi, \
        y_train_pred, y_val_pred, tot_t = train_val_test.train_model(model=model, 
                            train_data=x_train_batches, train_labels=y_train_batches,
                            loss_name=loss_fn, optimizer=optimizer,
                            epochs=epochs, early_stopping=early_stopping,
                            val_data=x_val_batches, val_labels=y_val_batches,
                            path_saving=path_saving_subfolder, path_tb=None)

    # SAVE CURRENT COMBINATION'S RESULTS
    # plot and save resulting losses
    plotting.losses_plot(train_losses=train_losses, val_losses=val_losses,
            loss_fn=loss_fn, display=False, save=True, path_saving=path_saving_subfolder)

    # plot ground truth vs predicted last wdw
    plotting.predicted_wdw_plot(x=x_val_batches, y=y_val_batches, 
                                y_pred=y_val_pred, wdw_nr=-1, last_pred=False,
                                display=True, save=True, path_saving=path_saving_subfolder)
    try:
        # plot ground truth vs predicted other wdw
        plotting.predicted_wdw_plot(x=x_val_batches, y=y_val_batches, 
                                    y_pred=y_val_pred, wdw_nr=-17, last_pred=False,
                                    display=True, save=True, path_saving=path_saving_subfolder)
        # plot ground truth vs predicted other wdw
        plotting.predicted_wdw_plot(x=x_val_batches, y=y_val_batches, 
                                    y_pred=y_val_pred, wdw_nr=-18, last_pred=False,
                                    display=True, save=True, path_saving=path_saving_subfolder)
    except IndexError:
        print('Attention: some predicted windows could not be plotted.')
        
    # save some stats for current set of net params
    full_net_params = f'net={net}-hidden_size={hidden_size}' + \
                f'-input_size={input_size}-epochs={epochs}' + \
                f'-loss_fn={loss_fn}-bi={bi} \n' + current_net_params
    utils.save_stats_train(path_saving_subfolder, full_net_params, other_params, 
                        loss_fn, train_losses, val_losses, val_losses_poi, tot_t) 
    
    # get best val loss poi for current combination and write it to overall list
    current_best_val_loss_poi = np.min(val_losses_poi)
    if loss_fn == 'MSE':
        current_best_val_RMSE_poi = np.sqrt(current_best_val_loss_poi)
    else:
        raise Exception('Attention: other loss functions not implemented')
    list_params[combination]['val_RMSE_poi'] = current_best_val_RMSE_poi

# keep track of ending time 
t1 = time.time()
# %%

# SAVE OVERALL GRID SEARCH RESULTS 
overall_best = 100000    
# save overall grid search result list
with open(os.path.join(path_saving, 'grid_search_output_list.txt'), 'w') as f:
    for combination in range(len(list_params)):
        print(list_params[combination], file=f)
        if list_params[combination]['val_RMSE_poi'] < overall_best:
            overall_best = list_params[combination]['val_RMSE_poi']
            overall_best_net_params = list_params[combination]
    print('\n', file=f)
    print('------- Overall best network hyper-parameters --------', file=f)
    print(overall_best_net_params, file=f)
    print('\n', file=f)
    print('------- Other parameters ---------', file=f)
    print(other_params, file=f)
    
    # timings
    print('\n', file=f)
    print(f'-------- Total time needed for grid search: {round((t1-t0)/60)} min --------- ', file=f) 
    