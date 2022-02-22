"""
Created on May 14 2021

@author: Elia Lombardo

Configuration file for main scripts
"""

# %%
# path inside docker container where project is mounted 
# or path to local folder where project was saved (to be adapted to your project)
path_project = '/home/centroid_prediction'

# path to data folder 
path_data = path_project + '/data'

# path to code folder
path_code = path_project + '/code'

# path to results folder
path_results = path_project + '/results'

import os
import matplotlib
import torch
import pandas as pd
import numpy as np

# add project code folder to your Python path to be able to import self written modules
import sys
sys.path.append(path_code)

# importing self written module from auxiliary folder 
from auxiliary import utils, data_preparation

# %%
# SET SCRIPT TO BE RUN AND GENERAL SETTINGS 

# whether to run data preprocessing or inference
code = 'predict'  # 'preprocess' or 'predict'

# whether to use GPU
gpu_usage = False
if gpu_usage:
    if torch.cuda.is_available():  
        device = torch.device('cuda:1')
        # set device nr to standard GPU
        torch.cuda.set_device(1)   # 0,1,2
    else:  
        device = torch.device('cpu') 
else:
    device = torch.device('cpu')
    
# print some running environment info
print_info_env = False
if print_info_env:
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION', )
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())
    print('\n')
    
    
# whether to run without display
no_display = False
if no_display:
    # to run on cluster with no display --> needs to be deactivated to display images
    matplotlib.use('agg')

# %%
# SET PARAMETERS FOR SCRIPTS

if code == 'preprocess':
    raise Exception('Not available. If interested write an email to Elia.Lombardo@med.uni-muenchen.de')

    
elif code == 'predict':
    # optimization phase
    phase = 'train'  # 'train' 'test'
    # actual data set used for inference, 
    # 'train', 'val' or 'test' can be used for sanity checks or real validation/testing
    # example: you have an offline trained model but want to check its perfomance 
    # again on the validation set but on a video basis like for online training
    # --> select phase='test' and set = 'val'
    set = 'val'  
    # relative path to folder with preprocessed data to be predicted
    cohort = '2021_06_16_respiratory_patients_LMU_ogv-no_norm'
    # cohort = '2021_10_25_free_breathing_patients_co60_Gemelli-no_norm'
    # total number of cases to be predicted
    nr_cases = len(utils.subdir_paths(os.path.join(path_data, 'preprocessed', cohort)))
    print(f'Number of cases found under "{cohort}": {nr_cases}')
    # relative amout train val and test cases
    train_split = 0.6
    val_split = 0.2
    test_split = round(1.0 - val_split - train_split, 2)
    print(f'Percentage of train, val, test cases: {train_split}, {val_split}, {test_split}')
    # aboslute amount of train val and test cases
    train_cases = round(train_split * nr_cases)
    train_val_cases = round(train_cases + val_split * nr_cases)
    
    # which type of normalization to apply on curves
    normalization = 'case_based'
    # motion direction to predict
    direction = 'SI'  # 'SI' 'AP'
    # whether to input motion with or without breathholds
    breathhold_inclusion = True
    # whether to input outlier replace or filterd and outlier replaced motion curves
    curve_aspect = 'f_or'  # 'or' 'f_or' 
    # total duration of set of sliding windows for online training
    min_train_data_length = 80  # 20 seconds 
    # length of input and output for windows of data snippets
    wdw_size_i = 32  # 8, 16, 24, 32 --> 2, 4, 6, 8 seconds
    wdw_size_o = 3  # 1, 2, 3 --> 250, 500, 750 ms
 
 
    # print some data info
    print_info = True
    if print_info:
        print(f'Normalization: {normalization}')
        print(f'Motion direction: {direction}')
        print(f'Breath-hold data inclusion: {breathhold_inclusion}')
        print(f'Curve aspect: {curve_aspect}')
        print(f'Input window size: {wdw_size_i}')
        print(f'Output window size: {wdw_size_o}')
        print('\n')

    # --- network and optimization parameters --------
    net = 'LSTM_stateless'  # LSTM_stateless, LR_closed_form
    offline_training = True  # if True, train first and then validate on a separate set   
    print(f'Model: {net}')
    print(f'Offline training: {offline_training}')
    print('\n') 
    
    
    path_results_train = os.path.join(path_results, net, 'train')
    # full path to offline trained model
    path_trained_model = None  # online validation/testing without pre trained model
    # path_trained_model = os.path.join(path_results_train, '2021-09-03-10:18:27_grid_search/num_layers=5-batch_size=64-lr=0.0005-l2=0-dropout=0')
    # path_trained_model = os.path.join(path_results_train, '2021-09-13-06:52:33_grid_search/num_layers=5-batch_size=64-lr=0.0005-l2=0-dropout=0')
    # path_trained_model = os.path.join(path_results_train, '2021-09-03-10:16:29_grid_search/num_layers=3-batch_size=64-lr=0.0005-l2=0-dropout=0/')
    # path_trained_model = os.path.join(path_results_train, '2021-09-03-10:11:28_grid_search/num_layers=5-batch_size=128-lr=0.0005-l2=0-dropout=0/')
    # path_trained_model = os.path.join(path_results_train, '2021-09-03-10:17:56_grid_search/num_layers=5-batch_size=64-lr=0.0005-l2=0-dropout=0/') 
    # path_trained_model = os.path.join(path_results_train, '2021-09-03-10:18:27_grid_search/num_layers=5-batch_size=64-lr=0.0005-l2=0-dropout=0/') 
    # path_trained_model = os.path.join(path_results_train, '2021-09-21-12:15:20/')
    # path_trained_model = os.path.join(path_results_train, '2021-09-29-07:22:12') 
    # path_trained_model = os.path.join(path_results_train, '2021-09-29-07:26:43')
    # path_trained_model = os.path.join(path_results_train, '2021-09-27-09:02:37') 

      
    grid_search = False
    if grid_search is False:
        
        if net == 'LSTM_stateless':
            # number of features for each time step
            input_size = 1
            # number of features of LSTM hidden state
            hidden_size = 15
            # number of LSTM hidden layers
            num_layers = 5
            # dropout probability on the outputs of each LSTM layer except the last layer
            dropout = 0.0
            # L2 regularization
            l2 = 1e-6
            # learning rate
            lr = 1e-6
            # nr of windows to be fed to network before updating weights 
            batch_size = 64           
            # loss function name
            loss_fn = 'MSE'  # 'MSE' 
            # bi-directionality of lstm
            bi = False
                  
        if net == 'LR_closed_form':
            # loss function name
            loss_fn = 'MSE'  # 'MSE'
            # L2 regularization (i.e. ridge regression)
            l2 = 1e-5
            # solver for sklearn LR model, use cholesky for closed form solution
            solver = 'cholesky'  # 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
            
        if offline_training:
            # nr of epochs for itertive models
            epochs = 600
            # early stop training if loss does not get better after early_stopping epochs
            early_stopping = epochs // 6  # None, int
            
        if offline_training is False:
            # nr of online training epochs fo iterative models
            online_epochs = 10

    # grid search
    else:
        # grid search ranges
        if net == 'LSTM_stateless':
            # number of features for each time step
            input_size = 1
            # number of features of LSTM hidden state
            hidden_size = 15
            # L2 regularization
            l2 = [0] 
            # number of LSTM hidden layers
            num_layers = [3, 5] 
            # num_layers = [3] 
            # dropout probability on the outputs of each LSTM layer except the last layer
            dropout = [0, 0.2] 
        # nr of windows to be fed to network before updating weights 
        batch_size = [64, 128] 
        # batch_size = [128] 
        # learning rate
        # lr = [0.0005, 0.001, 0.005] 
        lr = [0.0005, 0.001]
        # lr = [0.01] 
        # nr of epochs
        epochs = 600
        # loss function name
        loss_fn = 'MSE'  # 'MSE'
        # early stop training if loss does not get better after early_stopping epochs
        early_stopping = epochs // 6  # None, int
        # bi-directionality of lstm
        bi = False
               
              
    if phase == 'train':
        # train models on multiple videos in training set and validate on independent validation set
        if offline_training:         
            # load training snippets
            x_train_snippets, y_train_snippets = data_preparation.load_data_snippets(
                                                    path_data=os.path.join(path_data, 
                                                    'preprocessed', cohort), 
                                                    train_cases=train_cases, 
                                                    train_val_cases=train_val_cases, 
                                                    direction=direction, curve_aspect=curve_aspect, 
                                                    breathhold_inclusion=breathhold_inclusion,
                                                    normalization=normalization,
                                                    wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o,
                                                    step_size=1,
                                                    save_min_max_amplitudes=True,  # to generate amplitudes for later val/test
                                                    phase='train')
            
            # load validation snippets
            x_val_snippets, y_val_snippets = data_preparation.load_data_snippets(
                                                    path_data=os.path.join(path_data, 
                                                    'preprocessed', cohort), 
                                                    train_cases=train_cases, 
                                                    train_val_cases=train_val_cases, 
                                                    direction=direction, curve_aspect=curve_aspect, 
                                                    breathhold_inclusion=breathhold_inclusion,
                                                    normalization=normalization,
                                                    wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o,
                                                    step_size=1,
                                                    save_min_max_amplitudes=True,  # to generate amplitudes for later val/test                                             
                                                    phase='val')
            
            # load testing snippets (only first time to get the normalization amplitudes 
            # by setting save_min_max_amplitudes=True)
            # x_test_snippets, y_test_snippets = data_preparation.load_data_snippets(
            #                                         path_data=os.path.join(path_data, 
            #                                         'preprocessed', cohort), 
            #                                         train_cases=train_cases, 
            #                                         train_val_cases=train_val_cases, 
            #                                         direction=direction, curve_aspect=curve_aspect, 
            #                                         breathhold_inclusion=breathhold_inclusion,
            #                                         normalization=normalization,
            #                                         wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o,
            #                                         step_size=1,
            #                                         save_min_max_amplitudes=True,  # to generate amplitudes for later val/test                                             
            #                                         phase='test')    
        
        # online training, i.e. (optionally) train and validate/test on sliding windows of data
        if offline_training is False:
            # load cine-video subdiveded train-val snippets
            x_snippets_videos = [] 
            y_snippets_videos = [] 
            
            # get path to cine videos of train-val patients
            df_info_videos = pd.read_excel(os.path.join(path_data, 
                                                    'preprocessed', cohort, \
                                                    'min_max_amplitudes_' + set + \
                                                    '_BH_' + str(breathhold_inclusion) + '_' + \
                                                    curve_aspect + '_' + direction + '_LMU.xlsx'),
                                          engine='openpyxl')
            path_videos = df_info_videos['File name'] 
            min_amplitudes = df_info_videos['Min amplitude [mm]'] 
            max_amplitudes = df_info_videos['Max amplitude [mm]'] 
            
            # loop over all val traces and get windows of snippets
            for path_video in path_videos:
                xy_snippets_current_video = data_preparation.load_data_snippets(
                                                    path_data=path_video,  
                                                    direction=direction, curve_aspect=curve_aspect, 
                                                    breathhold_inclusion=breathhold_inclusion,
                                                    normalization=normalization,
                                                    wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o,
                                                    step_size=1, 
                                                    specific_patient=True)
                
                x_snippets_videos.append(xy_snippets_current_video[0])  
                y_snippets_videos.append(xy_snippets_current_video[1]) 
                                                                                      

    if phase == 'test':
        # load cine-video subdiveded train-test snippets
        x_snippets_videos = [] 
        y_snippets_videos = [] 
        
        # get amplitudes for normalization 
        # only first time: excel needs to be generated via load_data_snippets_function
        df_info_videos = pd.read_excel(os.path.join(path_data, 
                                                'preprocessed', cohort, \
                                                'min_max_amplitudes_' + set + \
                                                '_BH_' + str(breathhold_inclusion) + '_' + \
                                                curve_aspect + '_' + direction + '.xlsx'),
                                        engine='openpyxl')
        path_videos = df_info_videos['File name'] 
        min_amplitudes = df_info_videos['Min amplitude [mm]'] 
        max_amplitudes = df_info_videos['Max amplitude [mm]'] 
        
        
        # loop over all val traces and get windows of snippets
        for path_video in path_videos:
            xy_snippets_current_video = data_preparation.load_data_snippets(
                                                path_data=path_video,  
                                                direction=direction, curve_aspect=curve_aspect, 
                                                breathhold_inclusion=breathhold_inclusion,
                                                normalization=normalization,
                                                wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o,
                                                step_size=1, 
                                                save_min_max_amplitudes=True,
                                                specific_patient=True)
            
            x_snippets_videos.append(xy_snippets_current_video[0])  
            y_snippets_videos.append(xy_snippets_current_video[1]) 
                
            
# %%
