"""
Created on May 14 2021

@author: Elia Lombardo

Useful functions
"""
# %%
import numpy as np
import torch
import cv2
import os
from skimage.morphology import closing, square
from skimage.measure import label
import pandas as pd
import matplotlib.pyplot as plt

# import self written modules
from auxiliary import metrics


# %%


def subdir_paths(path):
    " Given a path the function returns only primary subdirectories in a sorted list. "
    return list(filter(os.path.isdir, [os.path.join(path, f) for f in sorted(os.listdir(path))]))


def normalize(values, bounds, single_value=False, to_tensor=False):
    """ Normalize values in range define by bounds.

    Args:
        values (list or array): data to be normalized, shape=(nr_data_points, 1)
        bounds (dict): current and desired bounds, for example
        {'actual':{'lower':5,'upper':15},'desired':{'lower':-1,'upper':1}}
        single_value: to give a single value as input (and output), i.e. nr_data_points=1
        to_tensor: convert to tensor

    Returns:
        array: array with normalized values
    """
    if single_value:
        print(f'values: {values}')
        if to_tensor:
            return torch.tensor(bounds['desired']['lower'] + (values - bounds['actual']['lower']) * \
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / \
                    (bounds['actual']['upper'] - bounds['actual']['lower']), requires_grad=True)             
        else:
            return bounds['desired']['lower'] + (values - bounds['actual']['lower']) * \
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / \
                    (bounds['actual']['upper'] - bounds['actual']['lower'])      
    else:  
        if to_tensor: 
            return np.array([torch.tensor(bounds['desired']['lower'] + (x - bounds['actual']['lower']) *
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / 
                    (bounds['actual']['upper'] - bounds['actual']['lower']), requires_grad=True) for x in values])
        else:
            return np.array([bounds['desired']['lower'] + (x - bounds['actual']['lower']) *
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / 
                    (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values])


def get_total_duration(wdws_of_snippets, wdw_size_i, step_size, fps=4):
    """ Get the total duration of a set of data windows 
    taking into account the overlap between them.

    Args:
        wdws_of_snippets (list of arrays): windows obtained from different snippets
        wdw_size_i (int): length of generated window to be used as input
        step_size (int): number of data points the window rolls at each step
        fps (int, optional): Frames per seconf of original video. Defaults to 4.
    """
    nr_time_points_all = [] 
    nr_time_points_to_subtract = [] 
    for wdws_of_snippet in wdws_of_snippets:
        nr_time_points_all.append(np.shape(wdws_of_snippet)[0] * \
                                    np.shape(wdws_of_snippet)[1])
        nr_time_points_to_subtract.append((np.shape(wdws_of_snippet)[0] - 1) * \
                                            (wdw_size_i - step_size))

    nr_time_points_eff = np.array(nr_time_points_all) - \
                            np.array(nr_time_points_to_subtract) 
                            
    total_duration = np.sum(nr_time_points_eff) / fps
    print(f'{round(total_duration, 2)} [s]; \
            {round(total_duration/60, 2)} [min]; \
            {round(total_duration/3600, 2)} [h]')
    
    
def save_stats_train_val(path_saving, net_params, other_params, loss_fn, 
               train_losses, val_losses, val_losses_poi, tot_t=0, unit='min'):
    """Save statistics generated during offline network training and validation to txt file.

    Args:
        path_saving (string): path to results folder
        net_params (string): network parameters used during optimization
        other_params (string): other parameters used 
        loss_fn (string): name of loss function used
        train_losses (list): training losses for different epochs
        val_losses (list): validation losses for different epochs
        val_losses_poi (list): validation losses for different epochs 
                        for prediction of interest, i.e. last time point
        tot_t (int): total time needed for optimization in 'unit'
    """

    with open(os.path.join(path_saving, 'stats.txt'), 'a') as file:
        file.write(f'Network parameters used: \n {net_params} \n')
        file.write(f'Other parameters used: \n {other_params} \n')
        file.write(f'Best train {loss_fn} = {np.min(train_losses)} \n')
        file.write(f'Best train RMSE = {np.sqrt(np.min(train_losses))} \n')
        file.write(f'Best val {loss_fn} = {np.min(val_losses)} \n')
        file.write(f'Best val RMSE = {np.sqrt(np.min(val_losses))} \n')
        file.write(f'Best val {loss_fn} poi = {np.min(val_losses_poi)} \n')
        file.write(f'Best val RMSE poi = {np.sqrt(np.min(val_losses_poi))} \n')
        file.write('\n')
        file.write(f'------ Total time needed for optimization: {tot_t} {unit} ------- ') 
    
    np.savetxt(os.path.join(path_saving, 'losses_train.txt'), train_losses)
    np.savetxt(os.path.join(path_saving, 'losses_val.txt'), val_losses)  
      

def save_stats_train_val_online(path_saving, net_params, other_params, 
                            y_pred_videos, y_batch_videos,
                            tot_times_online=0, 
                            set='val', info=''):
    """Save statistics generated during online network training (optional) and validation/testing to txt file.

    Args:
        path_saving (string): path to results folder
        net_params (string): network parameters used during optimization
        other_params (string): other parameters used 
        y_pred_videos (list of list with Pytorch tensors or np.arrays): predicted output series        
        y_batch_videos (list of list with Pytorch tensors or np.arrays): ground truth output series
        tot_times_online (list): list with online training times                           
        set (str): either 'val' or 'test', gives info on which data set was actually used for model inference
        info (str): additional info that can be added to txt file name
    """
    
    mses = []
    rmses = []
    mes = []   
    for y_pred_case, y_batch_case in zip(y_pred_videos, y_batch_videos): 
        if torch.is_tensor(y_pred_case[0]):
            # get tensor out of list of tensors
            y_pred = torch.stack(y_pred_case)
            y_batch = torch.stack(y_batch_case)
            # get numpy arrays on CPU
            y_pred = y_pred.detach().cpu().numpy()
            y_batch = y_batch.detach().cpu().numpy()
        elif isinstance(y_pred_case[0], np.ndarray):
            # get array out of list of arrays
            y_pred = np.concatenate(y_pred_case)
            y_batch = np.concatenate(y_batch_case)
        else:
            print(y_pred_case)
            raise Exception('Attention: unknown type')

        mses.append(metrics.MSE_array(y_pred, y_batch))
        rmses.append(metrics.RMSE_array(y_pred, y_batch))   
        mes.append(metrics.ME_array(y_pred, y_batch))
        
    # save rmses and mes
    np.savetxt(os.path.join(path_saving, 'rmses' + info + '.txt'), rmses)
    np.savetxt(os.path.join(path_saving, 'mes' + info + '.txt'), mes)
    
    # average over all videos and get error
    average_mse = np.mean(mses)
    average_rmse = np.mean(rmses)
    average_me = np.mean(mes)
    std_mse = np.std(mses)
    std_rmse = np.std(rmses)
    std_me = np.std(mes)    
    # get training timings
    average_tot_t_online = np.mean(tot_times_online) 

    with open(os.path.join(path_saving, 'stats' + info + '.txt'), 'a') as file:
        file.write(f'Network parameters used: \n {net_params} \n')
        file.write(f'Other parameters used: \n {other_params} \n')
        file.write(f'Average {set} MSE = {average_mse} \n')
        file.write(f'STD {set} MSE = {std_mse} \n')
        file.write(f'Average {set} RMSE = {average_rmse} \n')
        file.write(f'STD {set} RMSE = {std_rmse} \n')
        file.write(f'Average {set} ME = {average_me} \n')
        file.write(f'STD {set} ME  = {std_me} \n')             
        file.write('\n')
        file.write(f'------ Average total time needed for online training: {average_tot_t_online} ms ------- ') 


def get_info_preprocessed_cohort(path_cohort, path_saving, save=False):
    """Get information from the excel sheets of a preprocessed data set.

    Args:
        path_cohort (str): path to preprocessed cohort
        path_saving (str): path to results folder to contain plots etc
        save (bool, optional): whether to save plots or not. Defaults to False.
    """
    
    snippet_durations_with_bh = [] 
    snippet_durations_without_bh = []
    nr_xlsx = 0
    
    # create folder for results
    os.makedirs(path_saving, exist_ok=True)
        
    # loop over all paths of cases 
    for path_case in subdir_paths(path_cohort):
        
        # eg 'case03'
        current_case = os.path.basename(path_case)
        print(f'\n-------- Current case: {current_case} -----------\n')  

        # loop over all files of one case
        for _dir_name, _subdir_list, file_list in os.walk(path_case):
            for file_name in file_list:
                
                # check if file is xlsx, else go to next file
                if file_name.endswith('.xlsx'):
                    nr_xlsx += 1
                    
                    # read in excel file for one video
                    df = pd.read_excel(os.path.join(path_case, file_name),
                                       engine='openpyxl')
                    
                    #print(df)
                    # append average snippets duration for each video
                    snippet_durations_without_bh.append(df.loc[0, 'Mean snippet duration without/with BH [s]'])
                    snippet_durations_with_bh.append(df.loc[1, 'Mean snippet duration without/with BH [s]'])
                    
    #  get video and case averaged mean snippets duration 
    mean_snippet_without_bh = np.mean(np.array(snippet_durations_without_bh))              
    mean_snippet_with_bh = np.mean(np.array(snippet_durations_with_bh)) 
             
    #  plot histograms
    plt.hist(snippet_durations_without_bh, bins=50, label=f'Mean: {round(mean_snippet_without_bh, 2)} s')   
    plt.legend(loc="upper right")
    plt.ylabel('Occurrence')
    plt.xlabel('Average snippet duration without BHs')
    if save:
        plt.savefig(os.path.join(path_saving, f'hist_snippet_duration_without_BHs.png'), 
                    bbox_inches="tight")
    plt.close()

    plt.hist(snippet_durations_with_bh, bins=50, label=f'Mean: {round(mean_snippet_with_bh, 2)} s')   
    plt.legend(loc="upper right")
    plt.ylabel('Occurrence')
    plt.xlabel('Average snippet duration with BHs')
    if save:
        plt.savefig(os.path.join(path_saving, f'hist_snippet_duration_with_BHs.png'), 
                    bbox_inches="tight")
    plt.close()  
    
    
    with open(os.path.join(path_saving, 'results_info.txt'), 'a') as file:
        file.write(f'Number of excel sheets found: {nr_xlsx} \n')
        
    print('...results saved.')
    
    
# %%
# TESTING OUT CODE

# get_info_preprocessed_cohort(path_cohort='/data/preprocessed/2021_06_16_respiratory_patients_LMU_ogv-no_norm/', 
#                          path_saving='/home/centroid_prediction/results/various/2021_06_16_respiratory_patients_LMU_ogv-no_norm/', 
#                          save=True)      
# %%
