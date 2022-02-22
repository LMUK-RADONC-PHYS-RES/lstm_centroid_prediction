"""
Created on October 2021

@author: Elia Lombardo

Main script to generate png with input and output windows for GIF
"""

# %%
import config

# %%

import pickle
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

# %%
# set paths to curves generated with model
# and to ground truth curves
cohort = 'LMU_without_BH'  # LMU_without_BH, Gemelli, LMU_with_BH

# LSTM
if cohort == 'LMU_without_BH':
    # path_curves = config.path_results + '/LSTM_stateless/test/2021-10-20-07:09:00_online'  # 250
    path_curves = config.path_results + '/LSTM_stateless/test/2021-10-11-07:54:05_online'  # 500
    # path_curves = config.path_results + '/LSTM_stateless/test/2021-10-20-07:05:49_online'  # 750

    video_nr = -2

if cohort == 'Gemelli':
    # path_curves = config.path_results + '/LSTM_stateless/test/2021-10-28-13:31:17_online'
    path_curves = config.path_results + '/LSTM_stateless/test/2021-10-28-13:32:12_online'
    # path_curves = config.path_results + '/LSTM_stateless/test/2021-10-28-13:33:02_online'

    # video_nr = -4  # peak amplitudes very different
    # video_nr = -5  # rather regular
    # video_nr = -6   # baseline drifts
    # video_nr = -7   # small breath holds
    video_nr = -8  # pretty regular

if cohort == 'LMU_with_BH':
    # path_curves = config.path_results + ''
    path_curves = config.path_results + '/LSTM_stateless/test/2021-10-27-12:31:24_online'
    # path_curves = config.path_results + ''
    
    # video_nr = -5 # BH end
    video_nr = -5
    



# load predicted curves model 
with open(os.path.join(path_curves, 'y_pred_videos.txt'), 'rb') as f:   
    loaded_data = pickle.load(f) 
    # get tensor out of list of tensors and get numpy arrays on CPU
    y_pred_videos = torch.stack(loaded_data[video_nr]).detach().cpu().numpy() 

# load ground-truth curves       
with open(os.path.join(path_curves, 'y_batch_videos.txt'), 'rb') as f: 
    loaded_data = pickle.load(f)   
    y_batch_videos = torch.stack(loaded_data[video_nr]).detach().cpu().numpy() 

print('...curves loaded!')        

# %%
# plot within a fixed time axis the input, true output and predicted output windows
path_case = f'various/GIFs/{cohort}/'

# set window sizes
wdw_size_i = 32  # 8, 16, 24, 32 --> 2, 4, 6, 8 seconds
wdw_size_o = 2  # 1, 2, 3 --> 250, 500, 750 ms

# create time axis
data_points = 20 * 4 
data_points_plus_output = data_points + wdw_size_o
t = np.arange(data_points_plus_output) / 4

filenames = [] 
offset_points = 30
for wdw_nr in range(offset_points, offset_points + data_points - wdw_size_i - 1):
    # get previous input points
    x_prev = y_batch_videos[offset_points:wdw_nr + 1]
    # get input window
    x = y_batch_videos[wdw_nr:wdw_nr + wdw_size_i]
    # get true output window
    y = y_batch_videos[wdw_nr + wdw_size_i:wdw_nr + wdw_size_i + wdw_size_o]
    # get predicted output window
    y_pred = y_pred_videos[wdw_nr + wdw_size_i:wdw_nr + wdw_size_i + wdw_size_o]
    
    # print(np.shape(x))  # (32, 1)
    # print(np.shape(y))  # (2, 1)
    # print(np.shape(y_pred))  # (2, 1)

    plt.rcParams.update({'font.size': 26})   
    plt.figure(figsize=(10, 7))  
    
    axs = plt.axes()        
  
    # previous input points
    axs.plot(t[:wdw_nr + 1 - offset_points], x_prev, 'o-', color='grey')
    # current input
    axs.plot(t[wdw_nr - offset_points:wdw_nr + wdw_size_i - offset_points], x, 'o-', color='black', 
             label="True input")
    # true output
    axs.plot(t[wdw_nr + wdw_size_i - offset_points:wdw_nr + wdw_size_i + wdw_size_o - offset_points], y, 'o-', color='blue', 
             label="True output")
    # predicted output
    axs.plot(t[wdw_nr + wdw_size_i - offset_points:wdw_nr + wdw_size_i + wdw_size_o - offset_points], y_pred, 'd-', color='red', 
             label="Predicted output")    

    
    axs.set_ylabel(f"Normalized SI target centroid")
    axs.set_ylim(min(y_batch_videos[offset_points:offset_points + data_points - wdw_size_i - 1]) - 0.09, max(y_batch_videos[offset_points:offset_points + data_points - wdw_size_i - 1]) + 0.09);
    axs.set_xlabel("Time [s]")
    axs.set_xlim(-0.25, t[-1] + 0.25)
    
    # set legend and grid
    # axs.legend(bbox_to_anchor=(1.1, 0.95)) 
    axs.legend(loc=(1.04, 0.66))  
    axs.grid(linestyle='dashed')
    
    # set minor ticks on and ticks on both sides
    axs.minorticks_on()
    axs.xaxis.set_minor_locator(plt.MaxNLocator())
    axs.tick_params(labeltop=False, labelright=False)
    axs.tick_params(which='both', top=True, right=True)
    
    # create file name and append to list
    filename = f'{wdw_nr}.png'
    filenames.append(filename)
    
    plt.savefig(os.path.join(config.path_results, 
                             path_case, 'pngs', filename),
            bbox_inches="tight")
    

# %%
frames = []
# load each file into a list
for filename in filenames:
    frames.append(imageio.imread(os.path.join(config.path_results, 
                        path_case, 'pngs', filename)))

# save them as frames into a gif 
fps = 2
path_to_gif = os.path.join(config.path_results, 
                        path_case, f'sliding_prediction_fps_{fps}_LSTM_{os.path.basename(path_curves)}_video_nr_{video_nr}_offset_{offset_points}.gif')
imageio.mimsave(path_to_gif, frames, format='GIF', fps=fps)
print('GIF saved to disk!')
# %%
