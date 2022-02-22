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
if net == 'LR_closed_form':
    from config import l2, solver
    from sklearn.linear_model import Ridge
else:
    raise Exception('Attention: code is intended for LR with sklearn')
from config import wdw_size_i, wdw_size_o
from config import loss_fn, offline_training, path_trained_model
from config import phase, direction, breathhold_inclusion, curve_aspect
from config import normalization, cohort, train_split, val_split, test_split
from config import min_train_data_length
# %%
from auxiliary import plotting, utils
from auxiliary import train_val_test, data_preparation

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle

# %%
# PREPARE FOLDER FOR RESULTS

# set strings for results  
net_params = f'net={net}-l2={l2}-solver={solver}' + \
                f'-trained_model={path_trained_model}' 

other_params = f'device={device}-wdw_size_i={wdw_size_i}-wdw_size_o={wdw_size_o}' + \
                f'-step_size={1}-direction={direction}' + \
                f'-breathhold_inclusion={breathhold_inclusion}' + \
                f'\n -curve_aspect={curve_aspect}-normalization={normalization}-cohort={cohort}' + \
                f'\n -train_split={train_split}-val_split={val_split}-test_split={test_split}' + \
                f'\n -set={config.set}-offline_training={offline_training}'
         
# create folder for results   
if offline_training:
    path_saving = os.path.join(config.path_results, net, phase, time.strftime('%Y-%m-%d-%H:%M:%S')) 
else:
    path_saving = os.path.join(config.path_results, net, phase, time.strftime('%Y-%m-%d-%H:%M:%S_online')) 
os.makedirs(path_saving, exist_ok=True)

# %%
# PREPARE DATA FOR OPTIMIZATION

if path_trained_model is not None:
    # load previously trained model
    path_best_model = os.path.join(path_trained_model, 'trained_model.pkl')
    print(f'Path to best model of optimization: {path_best_model} \n')
    model = joblib.load(path_best_model)
    

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
                                                            to_tensor=False, 
                                                            gpu_usage=gpu_usage, device=device)
    y_batches = data_preparation.get_data_batches(data=y_snippets_videos[video_nr], 
                                                            batch_size=1, 
                                                            concat=True,
                                                            to_tensor=False,
                                                            gpu_usage=gpu_usage, device=device)
    print(f'Shape of obtained x_train_batches: {x_batches.shape}')  # (nr_batches, 1, wdw_size_i)
    print(f'Shape of obtained y_train_batches: {y_batches.shape}')  # (nr_batches, 1, wdw_size_o)

    # check if there is enough data in video to build set of training sliding windows of length 'train_data_duration'
    data_length_video = wdw_size_i + 1 * x_batches.shape[0] - 1
    if data_length_video > min_train_data_length + wdw_size_o:
        print(f'Number of data points for current video: {data_length_video}')
    
        # build model from scratch
        if path_trained_model is None:
            print('... proceeding with model initialization!')
            model = Ridge(alpha=l2, fit_intercept=True, solver=solver)

        # (if offline_traing=False) train (either pre-trained or newly initialized) model on 30 seconds   
        # of data subdivided in windows and then validate/test in a sliding window fashion
        train_online_losses_wdws, val_losses_wdws, val_losses_poi_wdws, \
                val_loss_video, val_loss_poi_video, \
                y_pred_video, y_batch_video, \
                tot_t_online = train_val_test.train_val_closed_LR_online(model=model, 
                                                        train_val_data=x_batches,
                                                        train_val_labels=y_batches, 
                                                        wdw_size_i=wdw_size_i,
                                                        wdw_size_o=wdw_size_o,
                                                        min_train_data_length=min_train_data_length,                                                        
                                                        offline_training=offline_training,
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
                                    single_value=False)))

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
    
    
    # plot and save losses as a function of wdws for last video
    plt.figure()
    plt.plot(val_losses_wdws)
    plt.savefig(os.path.join(path_saving, f'last_video_{config.set}_loss_vs_wdw_nr.png'))
    # plot train losses as a function of wdws for last video
    plt.figure()
    plt.plot(train_online_losses_wdws, '-o')
    plt.savefig(os.path.join(path_saving, 'last_video_train_online_loss_vs_wdw_nr.png'))

 
except IndexError:
    print('Expected IndexError occured when trying to plot results...')

# %%     