"""
Created on October 2021

@author: Elia Lombardo

Main script to compare ground truth vs predicted motion curves
"""

# %%
import config
from auxiliary import plotting

# %%

import pickle
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# %%
# SINGLE PLOT
# set paths to curves generated with different models
# and to ground truth curves

# LR
# path_curves_1 = config.path_results + '/LR_closed_form/test/2021-10-20-07:18:03'   # 250 ms
# path_curves_1 = config.path_results + '/LR_closed_form/test/2021-10-11-08:00:47'   # 500 ms
path_curves_1 = config.path_results + '/LR_closed_form/test/2021-10-20-07:15:20'   # 750 ms

# LSTM
# path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-20-07:09:00_online'
# path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-11-07:54:05_online'
path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-20-07:05:49_online'


# load predicted curves model 1
with open(os.path.join(path_curves_1, 'y_pred_videos.txt'), 'rb') as f:   
    y_pred_videos_1 = pickle.load(f) 
with open(os.path.join(path_curves_1, 'y_pred_videos_mm.txt'), 'rb') as f:   
    y_pred_videos_mm_1 = pickle.load(f) 

# load predicted curves model 2
with open(os.path.join(path_curves_2, 'y_pred_videos.txt'), 'rb') as f:   
    y_pred_videos_2 = pickle.load(f) 
with open(os.path.join(path_curves_2, 'y_pred_videos_mm.txt'), 'rb') as f:   
    y_pred_videos_mm_2 = pickle.load(f) 

# load ground-truth curves       
with open(os.path.join(path_curves_2, 'y_batch_videos.txt'), 'rb') as f:   
    y_batch_videos = pickle.load(f) 
with open(os.path.join(path_curves_2, 'y_batch_videos_mm.txt'), 'rb') as f:   
    y_batch_videos_mm = pickle.load(f)  

print('...curves loaded!')        
# %%     
# select snippet and range to plot
video_nr = -5
first_points = None
last_points = None
plotting.predicted_snippets_comparison(y_pred_1=y_pred_videos_mm_1[video_nr], 
                            y_pred_2=y_pred_videos_mm_2[video_nr], 
                            y_batch=y_batch_videos_mm[video_nr], 
                            video_nr=video_nr,
                            normalization=False, 
                            first_points=first_points, 
                            last_points=last_points,
                            display=True, save=True,
                            legend=False,
                            info=f'LR={os.path.basename(path_curves_1)}-LSTM={os.path.basename(path_curves_2)}', 
                            path_saving=os.path.join(config.path_results, 'various/compare_predicted_curves'))        


# %%
# plot difference between ground truth and predicted curves as violinplot
plotting.diff_violinplot(y_pred_1=y_pred_videos_mm_1, 
                         y_pred_2=y_pred_videos_mm_2,
                         y_batch=y_batch_videos_mm,
                            display=True, save=True, 
                            info=f'mm_LR={os.path.basename(path_curves_1)}-LSTM={os.path.basename(path_curves_2)}', 
                            path_saving=os.path.join(config.path_results, 'various/compare_predicted_curves'))   

# %%
# MULTIPLE PLOTS

# set paths to curves generated with different models
# and to ground truth curves

# LR
# (LMU)
path_curves_1_250 = config.path_results + '/LR_closed_form/test/2021-10-20-07:18:03'   # 250 ms
path_curves_1_500 = config.path_results + '/LR_closed_form/test/2021-10-11-08:00:47'   # 500 ms
path_curves_1_750 = config.path_results + '/LR_closed_form/test/2021-10-20-07:15:20'   # 750 ms
# (Gemelli)
# path_curves_1_250 = config.path_results + '/LR_closed_form/test/2021-10-28-13:19:14_online'
# path_curves_1_500 = config.path_results + '/LR_closed_form/test/2021-10-28-13:19:57_online'
# path_curves_1_750 = config.path_results + '/LR_closed_form/test/2021-10-28-13:20:30_online'


# LSTM
# (LMU)
path_curves_2_250 = config.path_results + '/LSTM_stateless/test/2021-10-20-07:09:00_online'
path_curves_2_500 = config.path_results + '/LSTM_stateless/test/2021-10-11-07:54:05_online'
path_curves_2_750 = config.path_results + '/LSTM_stateless/test/2021-10-20-07:05:49_online'
# (Gemelli)
# path_curves_2_250 = config.path_results + '/LSTM_stateless/test/2021-10-28-13:31:17_online'
# path_curves_2_500 = config.path_results + '/LSTM_stateless/test/2021-10-28-13:32:12_online'
# path_curves_2_750 = config.path_results + '/LSTM_stateless/test/2021-10-28-13:33:02_online'


video_nr_reg = -5
video_nr_irreg = -29
video_nr_irreg_250 = -31
# video_nr_reg = -2
# video_nr_irreg = -1
# video_nr_irreg_250 = -1
# load predicted curves model 1
with open(os.path.join(path_curves_1_250, 'y_pred_videos_mm.txt'), 'rb') as f:  
    loaded_data = pickle.load(f)
    y_pred_videos_mm_1_250_reg = np.concatenate(loaded_data[video_nr_reg])
    y_pred_videos_mm_1_250_irreg = np.concatenate(loaded_data[video_nr_irreg_250])
with open(os.path.join(path_curves_1_500, 'y_pred_videos_mm.txt'), 'rb') as f:  
    loaded_data = pickle.load(f) 
    y_pred_videos_mm_1_500_reg = np.concatenate(loaded_data[video_nr_reg])
    y_pred_videos_mm_1_500_irreg = np.concatenate(loaded_data[video_nr_irreg])
with open(os.path.join(path_curves_1_750, 'y_pred_videos_mm.txt'), 'rb') as f:   
    loaded_data = pickle.load(f) 
    y_pred_videos_mm_1_750_reg = np.concatenate(loaded_data[video_nr_reg])
    y_pred_videos_mm_1_750_irreg = np.concatenate(loaded_data[video_nr_irreg])
        
# load predicted curves model 2
with open(os.path.join(path_curves_2_250, 'y_pred_videos_mm.txt'), 'rb') as f:   
    loaded_data = pickle.load(f) 
    # get tensor out of list of tensors and get numpy arrays on CPU
    y_pred_videos_mm_2_250_reg = torch.stack(loaded_data[video_nr_reg]).detach().cpu().numpy() 
    y_pred_videos_mm_2_250_irreg = torch.stack(loaded_data[video_nr_irreg_250]).detach().cpu().numpy() 
with open(os.path.join(path_curves_2_500, 'y_pred_videos_mm.txt'), 'rb') as f:   
    loaded_data = pickle.load(f) 
    # get tensor out of list of tensors and get numpy arrays on CPU
    y_pred_videos_mm_2_500_reg = torch.stack(loaded_data[video_nr_reg]).detach().cpu().numpy() 
    y_pred_videos_mm_2_500_irreg = torch.stack(loaded_data[video_nr_irreg]).detach().cpu().numpy() 
with open(os.path.join(path_curves_2_750, 'y_pred_videos_mm.txt'), 'rb') as f:   
    loaded_data = pickle.load(f) 
    # get tensor out of list of tensors and get numpy arrays on CPU
    y_pred_videos_mm_2_750_reg = torch.stack(loaded_data[video_nr_reg]).detach().cpu().numpy() 
    y_pred_videos_mm_2_750_irreg = torch.stack(loaded_data[video_nr_irreg]).detach().cpu().numpy() 
    
# load ground-truth curves       
with open(os.path.join(path_curves_1_250, 'y_batch_videos_mm.txt'), 'rb') as f: 
    loaded_data = pickle.load(f)   
    y_batch_videos_mm_250_reg = np.concatenate(loaded_data[video_nr_reg])
    y_batch_videos_mm_250_irreg = np.concatenate(loaded_data[video_nr_irreg_250])
with open(os.path.join(path_curves_1_500, 'y_batch_videos_mm.txt'), 'rb') as f:   
    loaded_data = pickle.load(f)   
    y_batch_videos_mm_500_reg = np.concatenate(loaded_data[video_nr_reg])
    y_batch_videos_mm_500_irreg = np.concatenate(loaded_data[video_nr_irreg])
with open(os.path.join(path_curves_1_750, 'y_batch_videos_mm.txt'), 'rb') as f:   
    loaded_data = pickle.load(f)   
    y_batch_videos_mm_750_reg = np.concatenate(loaded_data[video_nr_reg])
    y_batch_videos_mm_750_irreg = np.concatenate(loaded_data[video_nr_irreg])
        
print('...curves loaded!')     

# %%
# make times for x axis
t = np.arange(len(y_batch_videos_mm_750_irreg)) / 4


# make huge subplot
fig, axs = plt.subplots(nrows=6, ncols=2, sharex=True, sharey=False, figsize=(27, 30), 
                        gridspec_kw={'height_ratios': [2, 1, 2, 1, 2, 1]})

# change fontsize for label, legend etc.
plt.rcParams.update({'font.size': 30})
plt.rcParams['axes.grid'] = True

# set minor ticks on and ticks on both sides
# for all axes
for row in range(6):
    for col in range(2):
        axs[row][col].minorticks_on()
        axs[row][col].xaxis.set_minor_locator(plt.MaxNLocator())
        axs[row][col].tick_params(labeltop=False, labelright=False)
        axs[row][col].tick_params(which='both', top=True, right=True)
    

# 250    
start = 6
stop = 63
# stop = 93
axs[0][0].plot(t[0:stop-start], y_batch_videos_mm_250_reg[start:stop], 'o-', color='black', label="True")
axs[0][0].plot(t[0:stop-start], y_pred_videos_mm_1_250_reg[start:stop], '*--', color='blue', label="LR")        
axs[0][0].plot(t[0:stop-start], y_pred_videos_mm_2_250_reg[start:stop], 'd--', color='red', label="LSTM")  
axs[0][0].set_ylabel('SI target centroid [mm]')
axs[0][0].annotate('(a)', xy=(-0.26, 0.94), xycoords='axes fraction', fontsize=32)
axs[0][0].annotate('250 ms', xy=(0.65, 0.11), xycoords='axes fraction')  # (0.65,0.14) Gemelli
diff_1_250_reg = y_batch_videos_mm_250_reg[start:stop] - y_pred_videos_mm_1_250_reg[start:stop]
diff_2_250_reg = y_batch_videos_mm_250_reg[start:stop] - y_pred_videos_mm_2_250_reg[start:stop, 0]
axs[1][0].plot(t[0:stop-start], diff_1_250_reg, '*--', color='blue')
axs[1][0].plot(t[0:stop-start], diff_2_250_reg, 'd--', color='red') 
axs[1][0].set_ylabel('Error [mm]')


start = 4
stop = 61
# stop = 91
axs[0][1].plot(t[0:stop-start], y_batch_videos_mm_250_irreg[start:stop], 'o-', color='black', label="True")
axs[0][1].plot(t[0:stop-start], y_pred_videos_mm_1_250_irreg[start:stop], '*--', color='blue', label="LR")        
axs[0][1].plot(t[0:stop-start], y_pred_videos_mm_2_250_irreg[start:stop], 'd--', color='red', label="LSTM")  
axs[0][1].legend(bbox_to_anchor=(1.32, 1.0))
diff_1_250_irreg = y_batch_videos_mm_250_irreg[start:stop] - y_pred_videos_mm_1_250_irreg[start:stop]
diff_2_250_irreg = y_batch_videos_mm_250_irreg[start:stop] - y_pred_videos_mm_2_250_irreg[start:stop, 0]
axs[1][1].plot(t[0:stop-start], diff_1_250_irreg, '*--', color='blue')
axs[1][1].plot(t[0:stop-start], diff_2_250_irreg, 'd--', color='red') 


# 500
start = 3
stop = 60
# stop = 90
axs[2][0].plot(t[0:stop-start], y_batch_videos_mm_500_reg[start:stop], 'o-', color='black', label="True")
axs[2][0].plot(t[0:stop-start], y_pred_videos_mm_1_500_reg[start:stop], '*--', color='blue', label="LR")        
axs[2][0].plot(t[0:stop-start], y_pred_videos_mm_2_500_reg[start:stop], 'd--', color='red', label="LSTM")  
axs[2][0].set_ylabel('SI target centroid [mm]')
axs[2][0].annotate('(b)', xy=(-0.26, 0.94), xycoords='axes fraction', fontsize=32)
axs[2][0].annotate('500 ms', xy=(0.65, 0.11), xycoords='axes fraction')  # () LMU
diff_1_500_reg = y_batch_videos_mm_500_reg[start:stop] - y_pred_videos_mm_1_500_reg[start:stop]
diff_2_500_reg = y_batch_videos_mm_500_reg[start:stop] - y_pred_videos_mm_2_500_reg[start:stop, 0]
axs[3][0].plot(t[0:stop-start], diff_1_500_reg, '*--', color='blue')
axs[3][0].plot(t[0:stop-start], diff_2_500_reg, 'd--', color='red') 
axs[3][0].set_ylabel('Error [mm]')

start = 2
stop = 59
# stop = 89
axs[2][1].plot(t[0:stop-start], y_batch_videos_mm_500_irreg[start:stop], 'o-', color='black', label="True")
axs[2][1].plot(t[0:stop-start], y_pred_videos_mm_1_500_irreg[start:stop], '*--', color='blue', label="LR")        
axs[2][1].plot(t[0:stop-start], y_pred_videos_mm_2_500_irreg[start:stop], 'd--', color='red', label="LSTM")  
diff_1_500_irreg = y_batch_videos_mm_500_irreg[start:stop] - y_pred_videos_mm_1_500_irreg[start:stop]
diff_2_500_irreg = y_batch_videos_mm_500_irreg[start:stop] - y_pred_videos_mm_2_500_irreg[start:stop, 0]
axs[3][1].plot(t[0:stop-start], diff_1_500_irreg, '*--', color='blue')
axs[3][1].plot(t[0:stop-start], diff_2_500_irreg, 'd--', color='red') 


# 750
start = 0
stop = 57
# stop = 87
axs[4][0].plot(t[0:stop-start], y_batch_videos_mm_750_reg[start:stop], 'o-', color='black', label="True")
axs[4][0].plot(t[0:stop-start], y_pred_videos_mm_1_750_reg[start:stop], '*--', color='blue', label="LR")        
axs[4][0].plot(t[0:stop-start], y_pred_videos_mm_2_750_reg[start:stop], 'd--', color='red', label="LSTM")  
axs[4][0].set_ylabel('SI target centroid [mm]')
axs[4][0].annotate('(c)', xy=(-0.26, 0.94), xycoords='axes fraction', fontsize=32)
axs[4][0].annotate('750 ms', xy=(0.65, 0.11), xycoords='axes fraction')  # () LMU
diff_1_750_reg = y_batch_videos_mm_750_reg[start:stop] - y_pred_videos_mm_1_750_reg[start:stop]
diff_2_750_reg = y_batch_videos_mm_750_reg[start:stop] - y_pred_videos_mm_2_750_reg[start:stop, 0]
axs[5][0].plot(t[0:stop-start], diff_1_750_reg, '*--', color='blue')
axs[5][0].plot(t[0:stop-start], diff_2_750_reg, 'd--', color='red') 
axs[5][0].set_ylabel('Error [mm]')
axs[5][0].set_xlabel('Time [s]')


start = 0
stop = 57
# stop = 87
axs[4][1].plot(t[0:stop-start], y_batch_videos_mm_750_irreg[start:stop], 'o-', color='black', label="True")
axs[4][1].plot(t[0:stop-start], y_pred_videos_mm_1_750_irreg[start:stop], '*--', color='blue', label="LR")        
axs[4][1].plot(t[0:stop-start], y_pred_videos_mm_2_750_irreg[start:stop], 'd--', color='red', label="LSTM")  
diff_1_750_irreg = y_batch_videos_mm_750_irreg[start:stop] - y_pred_videos_mm_1_750_irreg[start:stop]
diff_2_750_irreg = y_batch_videos_mm_750_irreg[start:stop] - y_pred_videos_mm_2_750_irreg[start:stop, 0]
axs[5][1].plot(t[0:stop-start], diff_1_750_irreg, '*--', color='blue')
axs[5][1].plot(t[0:stop-start], diff_2_750_irreg, 'd--', color='red') 
axs[5][1].set_xlabel('Time [s]')


fig.subplots_adjust(wspace=0.15, hspace=0.1)
fig.align_ylabels(axs[:, 0])
plt.savefig(os.path.join(config.path_results, 'various/compare_predicted_curves/predicted_snippets_all_LMU.png'),
            bbox_inches="tight")
# %%
