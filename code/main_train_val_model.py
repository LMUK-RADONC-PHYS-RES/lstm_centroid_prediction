"""
Created on May 25 2021

@author: Elia Lombardo

Main script to train and validate model for centroid position prediction in offline fashion
"""
# %%
# IMPORT MODULES

# self written modules & parameters from config
import config
if config.grid_search:
    raise Exception('Attention: grid search was set to True')
from config import gpu_usage
from config import device
from config import x_train_snippets, y_train_snippets
from config import x_val_snippets, y_val_snippets
from config import net
if net == 'LSTM_stateless':
    from config import input_size, hidden_size, num_layers, dropout, bi
from config import l2
from config import wdw_size_i, wdw_size_o
from config import lr, epochs, batch_size
from config import loss_fn, early_stopping
from config import phase, direction, breathhold_inclusion, curve_aspect
from config import normalization, cohort, train_split, val_split, test_split
from auxiliary import plotting, architectures, utils
from auxiliary import train_val_test, data_preparation

import time
import numpy as np
import os
import torch

# %%
# PREPARE FOLDER FOR RESULTS

# set strings for results
if net == 'LSTM_stateless':
    net_params = f'net={net}-hidden_size={hidden_size}-num_layers={num_layers}' + \
                    f'-input_size={input_size}-batch_size={batch_size}-lr={lr}-epochs={epochs}' + \
                    f'-l2={l2}-dropout={dropout}-loss_fn={loss_fn}-bi={bi}'             
         
other_params = f'device={device}-wdw_size_i={wdw_size_i}-wdw_size_o={wdw_size_o}' + \
                f'-step_size={1}-direction={direction}' + \
                f'-breathhold_inclusion={breathhold_inclusion}' + \
                f'\n -curve_aspect={curve_aspect}-normalization={normalization}-cohort={cohort}' + \
                f'\n -train_split={train_split}-val_split={val_split}-test_split={test_split}'
         
# create folder for results   
path_tb = os.path.join(config.path_results, 'tensorboard_runs')          
path_saving = os.path.join(config.path_results, net, phase, time.strftime('%Y-%m-%d-%H:%M:%S')) 
os.makedirs(path_saving, exist_ok=True)

# %%
# PREPARE DATA FOR OPTIMIZATION

# get batches of data
x_train_batches = data_preparation.get_data_batches(data=x_train_snippets, batch_size=batch_size, 
                                                    concat=True,
                                                    to_tensor=True, 
                                                    gpu_usage=gpu_usage, device=device)
y_train_batches = data_preparation.get_data_batches(data=y_train_snippets, batch_size=batch_size, 
                                                    concat=True,
                                                    to_tensor=True,
                                                    gpu_usage=gpu_usage, device=device)
print('\n')
print(f'Shape of obtained x_train_batches: {x_train_batches.shape}')  # (nr_batches, batch_size, wdw_size_i)
print(f'Shape of obtained y_train_batches: {y_train_batches.shape}')  # (nr_batches, batch_size, wdw_size_o)
print('\n')

x_val_batches = data_preparation.get_data_batches(data=x_val_snippets, batch_size=batch_size, 
                                                concat=True,
                                                to_tensor=True, 
                                                gpu_usage=gpu_usage, device=device)
y_val_batches = data_preparation.get_data_batches(data=y_val_snippets, batch_size=batch_size, 
                                                concat=True,
                                                to_tensor=True, 
                                                gpu_usage=gpu_usage, device=device)

# %%
# MODEL BUILDING AND OPTIMIZATION

if net == 'LSTM_stateless':
    # create LSTM instance
    model = architectures.CentroidPredictionLSTM(input_size=input_size, hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    seq_length_in=wdw_size_i, seq_length_out=wdw_size_o,
                                    dropout=dropout, bi=bi,
                                    gpu_usage=gpu_usage, device=device)
else:
    raise Exception('Attention: unknown net name!')


if gpu_usage:
    model.to(device)
print(model)

# get nr of trainable params
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of trainable parameters: {pytorch_total_params}\n')

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

# train (and validate) the model
train_losses, val_losses, val_losses_poi, \
    y_train_pred, y_val_pred, tot_t = train_val_test.train_model(model=model, 
                        train_data=x_train_batches, train_labels=y_train_batches,
                        loss_name=loss_fn, optimizer=optimizer,
                        epochs=epochs, early_stopping=early_stopping,
                        val_data=x_val_batches, val_labels=y_val_batches,
                        path_saving=path_saving, path_tb=None)

# %%
# SAVE RESULTS

# save some stats
utils.save_stats_train_val(path_saving, net_params, other_params, 
                    loss_fn, train_losses, 
                    val_losses, val_losses_poi,
                    tot_t) 

# plot and save resulting losses
plotting.losses_plot_detailed(train_losses=train_losses, val_losses=val_losses,
           loss_fn=loss_fn, log=True, display=False, 
           save=True, path_saving=path_saving)

# plot ground truth vs predicted last wdw
plotting.predicted_wdw_plot(x=x_val_batches[-1, ...], y=y_val_batches[-1, ...], 
                            y_pred=y_val_pred, wdw_nr=-1, last_pred=False,
                            display=True, save=True, path_saving=path_saving)
try:
    # plot ground truth vs predicted other wdw
    plotting.predicted_wdw_plot(x=x_val_batches[-1, ...], y=y_val_batches[-1, ...], 
                                y_pred=y_val_pred, wdw_nr=-7, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    # plot ground truth vs predicted other wdw
    plotting.predicted_wdw_plot(x=x_val_batches[-1, ...], y=y_val_batches[-1, ...], 
                                y_pred=y_val_pred, wdw_nr=-8, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    # plot ground truth vs predicted other wdw
    plotting.predicted_wdw_plot(x=x_val_batches[-1, ...], y=y_val_batches[-1, ...], 
                                y_pred=y_val_pred, wdw_nr=-27, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    # plot ground truth vs predicted other wdw
    plotting.predicted_wdw_plot(x=x_val_batches[-1, ...], y=y_val_batches[-1, ...], 
                                y_pred=y_val_pred, wdw_nr=-28, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    # plot ground truth vs predicted other wdw
    plotting.predicted_wdw_plot(x=x_val_batches[-1, ...], y=y_val_batches[-1, ...], 
                                y_pred=y_val_pred, wdw_nr=-29, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    # plot ground truth vs predicted other wdw
    plotting.predicted_wdw_plot(x=x_val_batches[-1, ...], y=y_val_batches[-1, ...], 
                                y_pred=y_val_pred, wdw_nr=-119, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    # plot ground truth vs predicted other wdw
    plotting.predicted_wdw_plot(x=x_val_batches[-1, ...], y=y_val_batches[-1, ...], 
                                y_pred=y_val_pred, wdw_nr=-120, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
except IndexError:
    print('Attention: some predicted windows could not be plotted.')
    

# %%
