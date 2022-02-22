"""
Created on May 25 2021

@author: Elia Lombardo

Main script to val/test model and if needed continuosly train model on single traces for centroid position prediction
"""
# %%
# IMPORT MODULES

# self written modules & parameters from config
import config

from config import gpu_usage
from config import device
from config import x_snippets_videos, y_snippets_videos, max_amplitudes, min_amplitudes
from config import net
if net == 'LSTM_stateless':
    from config import input_size, hidden_size, num_layers, dropout, l2, bi
from config import wdw_size_i, wdw_size_o
from config import lr
from config import loss_fn, offline_training, path_trained_model
from config import phase, direction, breathhold_inclusion, curve_aspect
from config import normalization, cohort, train_split, val_split, test_split
from config import min_train_data_length
# %%
from auxiliary import plotting, architectures, utils
from auxiliary import train_val_test, data_preparation

import time
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

# %%
# PREPARE FOLDER FOR RESULTS

# set strings for results
if net == 'LSTM_stateless':
    net_params = f'net={net}-hidden_size={hidden_size}-num_layers={num_layers}' + \
                    f'-input_size={input_size}-lr={lr}' + \
                    f'-l2={l2}-dropout={dropout}-loss_fn={loss_fn}-bi={bi}' + \
                    f'-trained_model={path_trained_model}'   


if offline_training:                                 
    other_params = f'device={device}-wdw_size_i={wdw_size_i}-wdw_size_o={wdw_size_o}' + \
                f'-step_size={1}-direction={direction}' + \
                f'-breathhold_inclusion={breathhold_inclusion}' + \
                f'\n -curve_aspect={curve_aspect}-normalization={normalization}-cohort={cohort}' + \
                f'\n -train_split={train_split}-val_split={val_split}-test_split={test_split}' + \
                f'\n -set={config.set}-offline_training={offline_training}'
if offline_training is False:
    other_params = f'device={device}-wdw_size_i={wdw_size_i}-wdw_size_o={wdw_size_o}' + \
                f'-step_size={1}-direction={direction}' + \
                f'-breathhold_inclusion={breathhold_inclusion}' + \
                f'\n -curve_aspect={curve_aspect}-normalization={normalization}-cohort={cohort}' + \
                f'\n -train_split={train_split}-val_split={val_split}-test_split={test_split}' + \
                f'\n -set={config.set}-offline_training={offline_training}' + \
                f'-online_epochs={config.online_epochs}'                
         
# create folder for results   
if offline_training:
    path_saving = os.path.join(config.path_results, net, phase, time.strftime('%Y-%m-%d-%H:%M:%S')) 
else:
    path_saving = os.path.join(config.path_results, net, phase, time.strftime('%Y-%m-%d-%H:%M:%S_online')) 
os.makedirs(path_saving, exist_ok=True)

# %%
# PREPARE DATA FOR OPTIMIZATION

if path_trained_model is not None:
    # load trained model
    model_files = [] 
    mses = [] 
    # loop over all subfolders and files of one pre training
    for dir_name, subdir_list, file_list in os.walk(path_trained_model):
        for file in file_list:
            print(file)
            if file[-4:] == '.pth':
                model_files.append(file)
                # append all the MSE values
                mses.append(float(file[-12:-4]))
                    
    mses = np.array(mses)
    model_files = np.array(model_files)

    # find best model by looking at the smallest MSE 
    best_model = model_files[np.argmin(mses)]
    path_best_model = os.path.join(path_trained_model, best_model)
    print(f'Path to best model of optimization: {path_best_model} \n')


    if net == 'LSTM_stateless':
        # create LSTM instance
        model = architectures.CentroidPredictionLSTM(input_size=input_size, hidden_size=hidden_size, 
                                        num_layers=num_layers, 
                                        seq_length_in=wdw_size_i, 
                                        seq_length_out=wdw_size_o,
                                        dropout=0, bi=bi,
                                        gpu_usage=gpu_usage, device=device)

    # load model weights
    model.load_state_dict(torch.load(path_best_model))  
    
    if gpu_usage:
        model.to(device)
    print(model)  


print(f'Number of videos: {len(x_snippets_videos)}')

val_losses_videos_wdws = [] 
val_losses_poi_videos_wdws = [] 
val_losses_videos = [] 
val_losses_poi_videos = [] 
y_pred_videos = [] 
y_batch_videos = [] 
y_pred_videos_mm = [] 
y_batch_videos_mm = [] 
tot_times_online = [] 
for video_nr in range(len(x_snippets_videos)):
    print('\n')
    print(f'----- Current video nr {video_nr}/{len(x_snippets_videos) - 1} ------- ')
    # plot input output windows for specific video
    if video_nr == 2:
        plot_path_saving = path_saving
    else:
        plot_path_saving = None
        
    # get amplitudes for current video to (later) undo the normalization
    min_amplitude = min_amplitudes[video_nr] 
    max_amplitude = max_amplitudes[video_nr]
    
    # get batches of data, i.e. for one video a list of tensors
    # of the windows extracted from (different) snippets
    x_batches = data_preparation.get_data_batches(data=x_snippets_videos[video_nr], 
                                                            batch_size=1, 
                                                            concat=True,
                                                            to_tensor=True, 
                                                            gpu_usage=gpu_usage, device=device)
    y_batches = data_preparation.get_data_batches(data=y_snippets_videos[video_nr], 
                                                            batch_size=1, 
                                                            concat=True,
                                                            to_tensor=True,
                                                            gpu_usage=gpu_usage, device=device)
    print(f'Shape of obtained x_batches: {x_batches.shape}')  # (nr_batches, 1, wdw_size_i)
    print(f'Shape of obtained y_batches: {y_batches.shape}')  # (nr_batches, 1, wdw_size_o)

    # check if there is enough data in video to build set
    # of training sliding windows of length 'train_data_duration'
    data_length_video = wdw_size_i + 1 * x_batches.shape[0] - 1
    if data_length_video > min_train_data_length + wdw_size_o:
        print(f'Number of data points for current video: {data_length_video}')
    
        # build model from scratch
        if path_trained_model is None:
            print('... proceeding with model initialization!')

            if net == 'LSTM_stateless':
                # create LSTM instance
                model = architectures.CentroidPredictionLSTM(input_size=input_size, 
                                                hidden_size=hidden_size, 
                                                num_layers=num_layers, 
                                                seq_length_in=wdw_size_i, 
                                                seq_length_out=wdw_size_o,
                                                dropout=dropout, bi=bi,
                                                gpu_usage=gpu_usage, device=device)

            if gpu_usage:
                model.to(device)
            print(model)
            
        # get nr of trainable params
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total number of trainable parameters: {pytorch_total_params}\n')

        # set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

        if offline_training:
            # validate/test in a sliding window fashion
            train_online_losses_wdws, \
                    val_losses_wdws, val_losses_poi_wdws, \
                    val_loss_video, val_loss_poi_video, \
                    y_pred_video, y_batch_video, \
                    tot_t_online = train_val_test.train_val_model_online(model=model, 
                                                        train_val_data=x_batches,
                                                        train_val_labels=y_batches, 
                                                        loss_name=loss_fn, 
                                                        optimizer=optimizer,
                                                        wdw_size_i=wdw_size_i,
                                                        wdw_size_o=wdw_size_o,
                                                        min_train_data_length=min_train_data_length,
                                                        online_epochs=None,
                                                        output_positions=True,
                                                        plot_path_saving=plot_path_saving)        
        if offline_training is False:
            # train (either pre-trained or newly initialized) model on 30 seconds 
            # of data subdivided in windows and validate/test in a sliding window fashion
            train_online_losses_wdws, \
                    val_losses_wdws, val_losses_poi_wdws, \
                    val_loss_video, val_loss_poi_video, \
                    y_pred_video, y_batch_video, \
                    tot_t_online = train_val_test.train_val_model_online(model=model, 
                                                        train_val_data=x_batches,
                                                        train_val_labels=y_batches, 
                                                        loss_name=loss_fn, 
                                                        optimizer=optimizer,
                                                        wdw_size_i=wdw_size_i,                                                        
                                                        wdw_size_o=wdw_size_o,
                                                        online_epochs=config.online_epochs,
                                                        min_train_data_length=min_train_data_length,
                                                        output_positions=True,
                                                        plot_path_saving=plot_path_saving)
                        
        
        # append prediction and labels for current video       
        y_pred_videos.append(y_pred_video)
        y_batch_videos.append(y_batch_video) 
               
        # undo normalization for predictions and labels
        y_pred_videos_mm.append(list(utils.normalize(y_pred_video, {'actual': {'lower': -1, 'upper': 1}, 
                                        'desired': {'lower': min_amplitude, 'upper': max_amplitude}},
                                    single_value=False)))
        y_batch_videos_mm.append(list(utils.normalize(y_batch_video, {'actual': {'lower': -1, 'upper': 1}, 
                                        'desired': {'lower': min_amplitude, 'upper': max_amplitude}},
                                    single_value=False, to_tensor=True)))

        # append training times
        tot_times_online.append(tot_t_online) 
        

# %%
# SAVE RESULTS

# save some stats   
utils.save_stats_train_val_online(path_saving, net_params, other_params, 
                    y_pred_videos, y_batch_videos,
                    tot_times_online,
                    set=config.set, info='')
utils.save_stats_train_val_online(path_saving, net_params, other_params, 
                    y_pred_videos_mm, y_batch_videos_mm,
                    tot_times_online,
                    set=config.set, info='_mm')

# save ground truth and predicted curves for later plotting etc
with open(os.path.join(path_saving, 'y_pred_videos.txt'), 'wb') as f:   
    pickle.dump(y_pred_videos, f) 
with open(os.path.join(path_saving, 'y_pred_videos_mm.txt'), 'wb') as f:   
    pickle.dump(y_pred_videos_mm, f) 
with open(os.path.join(path_saving, 'y_batch_videos.txt'), 'wb') as f:   
    pickle.dump(y_batch_videos, f) 
with open(os.path.join(path_saving, 'y_batch_videos_mm.txt'), 'wb') as f:   
    pickle.dump(y_batch_videos_mm, f) 
                  
try:
    # plot ground truth vs predicted snippets
    plotting.predicted_snippets_plot(y_pred=y_pred_videos[-1], y_batch=y_batch_videos[-1], 
                                    normalization=True, first_points=64,  # 16 s
                                    display=True, save=True, path_saving=path_saving)
    plotting.predicted_snippets_plot(y_pred=y_pred_videos_mm[-1], y_batch=y_batch_videos_mm[-1], 
                                    normalization=False, first_points=64,
                                    display=True, save=True, path_saving=path_saving)

    plotting.predicted_snippets_plot(y_pred=y_pred_videos[-1], y_batch=y_batch_videos[-1], 
                                    normalization=True, last_points=64,  
                                    display=True, save=True, path_saving=path_saving)
    plotting.predicted_snippets_plot(y_pred=y_pred_videos_mm[-1], y_batch=y_batch_videos_mm[-1], 
                                    normalization=False, last_points=64,
                                    display=True, save=True, path_saving=path_saving)
    
    # plot val/test losses as a function of wdws for last video
    plt.figure()
    plt.plot(val_losses_wdws)
    plt.savefig(os.path.join(path_saving, f'last_video_{config.set}_loss_vs_wdw_nr.png'))
    # plot train losses as a function of wdws for last video
    plt.figure()
    plt.plot(train_online_losses_wdws, '-o')
    plt.savefig(os.path.join(path_saving, 'last_video_train_online_loss_vs_wdw_nr.png'))

 
except IndexError:
    print('Expected IndexError occured when trying to plot some results...')

# %%
