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
if net == 'LR_closed_form':
    from config import l2, solver
    from sklearn.linear_model import Ridge    
else:
    raise Exception('Attention: code is programmed for LR with sklearn')
from config import wdw_size_i, wdw_size_o
from config import loss_fn
from config import phase, direction, breathhold_inclusion, curve_aspect
from config import normalization, cohort, train_split, val_split, test_split
# %%
from auxiliary import plotting, utils
from auxiliary import train_val_test, data_preparation

import time
import numpy as np
import os
# %%
# PREPARE FOLDER FOR RESULTS

# set strings for results
if net == 'LR_closed_form':
    net_params = f'net={net}-l2={l2}-solver={solver}-loss_fn={loss_fn}'                 
         
other_params = f'device={device}-wdw_size_i={wdw_size_i}-wdw_size_o={wdw_size_o}' + \
                f'-step_size={1}-direction={direction}' + \
                f'-breathhold_inclusion={breathhold_inclusion}' + \
                f'\n -curve_aspect={curve_aspect}-normalization={normalization}-cohort={cohort}' + \
                f'\n -train_split={train_split}-val_split={val_split}-test_split={test_split}'
         
# create folder for results   
path_saving = os.path.join(config.path_results, net, phase, time.strftime('%Y-%m-%d-%H:%M:%S')) 
os.makedirs(path_saving, exist_ok=True)

# %%
# PREPARE DATA FOR OPTIMIZATION

# get batches of data
x_train_batches = data_preparation.get_data_batches(data=x_train_snippets, batch_size=1, 
                                                    concat=True,
                                                    to_tensor=False, 
                                                    gpu_usage=gpu_usage, device=device)
y_train_batches = data_preparation.get_data_batches(data=y_train_snippets, batch_size=1, 
                                                    concat=True,
                                                    to_tensor=False,
                                                    gpu_usage=gpu_usage, device=device)
print('\n')
print(f'Shape of obtained x_train_batches: {x_train_batches.shape}')  # (nr_batches, 1, wdw_size_i)
print(f'Shape of obtained y_train_batches: {y_train_batches.shape}')  # (nr_batches, 1, wdw_size_o)
print('\n')

x_val_batches = data_preparation.get_data_batches(data=x_val_snippets, batch_size=1, 
                                                concat=True,
                                                to_tensor=False, 
                                                gpu_usage=gpu_usage, device=device)
y_val_batches = data_preparation.get_data_batches(data=y_val_snippets, batch_size=1, 
                                                concat=True,
                                                to_tensor=False, 
                                                gpu_usage=gpu_usage, device=device)

# %%
# MODEL BUILDING AND OPTIMIZATION

if net == 'LR_closed_form':
    model = Ridge(alpha=l2, fit_intercept=True, solver=solver)
else:
    raise Exception('Attention: unknown net name!')

# train (and validate) the model
train_loss, val_loss, val_loss_poi, \
    y_train_pred, y_val_pred, tot_t = train_val_test.train_closed_LR(model=model, 
                        train_data=x_train_batches, train_labels=y_train_batches,
                        val_data=x_val_batches, val_labels=y_val_batches,
                        path_saving=path_saving)

# %%
# SAVE RESULTS
print(f'model.coef_ {model.coef_}')
print(f'model.intercept_ {model.intercept_}')

# save some stats
utils.save_stats_train_val(path_saving, net_params, other_params, 
                    loss_fn, train_loss, 
                    val_loss, val_loss_poi,
                    tot_t, 'ms') 


# plot ground truth vs predicted last wdw
plotting.predicted_wdw_plot(x=x_val_batches[:, 0, :], y=y_val_batches[:, 0, :], 
                            y_pred=y_val_pred, wdw_nr=-1, last_pred=False,
                            display=True, save=True, path_saving=path_saving)
try:
    # plot ground truth vs predicted other wdw
    plotting.predicted_wdw_plot(x=x_val_batches[:, 0, :], y=y_val_batches[:, 0, :], 
                                y_pred=y_val_pred, wdw_nr=-7, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    plotting.predicted_wdw_plot(x=x_val_batches[:, 0, :], y=y_val_batches[:, 0, :], 
                                y_pred=y_val_pred, wdw_nr=-8, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    plotting.predicted_wdw_plot(x=x_val_batches[:, 0, :], y=y_val_batches[:, 0, :], 
                                y_pred=y_val_pred, wdw_nr=-27, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    plotting.predicted_wdw_plot(x=x_val_batches[:, 0, :], y=y_val_batches[:, 0, :], 
                                y_pred=y_val_pred, wdw_nr=-28, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    plotting.predicted_wdw_plot(x=x_val_batches[:, 0, :], y=y_val_batches[:, 0, :], 
                                y_pred=y_val_pred, wdw_nr=-29, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    plotting.predicted_wdw_plot(x=x_val_batches[:, 0, :], y=y_val_batches[:, 0, :], 
                                y_pred=y_val_pred, wdw_nr=-119, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
    plotting.predicted_wdw_plot(x=x_val_batches[:, 0, :], y=y_val_batches[:, 0, :], 
                                y_pred=y_val_pred, wdw_nr=-120, last_pred=False,
                                display=True, save=True, path_saving=path_saving)
except IndexError:
    print('Attention: some predicted windows could not be plotted.')
    

# %%
