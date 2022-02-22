import numpy as np
import torch
import torch.nn as nn


def MSE(data, labels):
    """
    Mean squared error metric based on torch MSE.
    Defaults to MSE = mean(L) where L is a vector containg 
    the following elements l_i = (data_i - labels_i)²
    
    Args:
        data (Pytorch tensor): predicted output 
        labels (Pytorch tensor): ground truth output   
    """
    
    metric = nn.MSELoss()
    metric_value = metric(data, labels)
    metric_value = metric_value.item()
    
    return metric_value

def RMSE(data, labels):
    """
    Root mean squared error metric based on torch MSE.
    
    Args:
        data (Pytorch tensor): predicted output 
        labels (Pytorch tensor): ground truth output   
    """
    
    metric = nn.MSELoss()
    metric_value = metric(data, labels)
    metric_value = np.sqrt(metric_value.item())
    
    return metric_value


def ME(data, labels):
    """
    Maximum error between predicted and true sequence.

    Args:
        data (list of Pytorch tensors): predicted outputs
        labels (list of Pytorch tensor): ground truth outputs    
    """
    # get tensor out of list of tensors
    data = torch.stack(data)
    labels = torch.stack(labels)
    
    # get numpy arrays on CPU
    data = data.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    
    metric_value = np.max(np.abs(data - labels))
    
    return metric_value


def MSE_array(data_array, labels_array):
    """
    Mean squared error with standard deviation between predicted and true sequence.

    Args:
        data (numpy array): predicted outputs
        labels (numpy array): ground truth outputs    
    """   
    difference_array = np.subtract(data_array, labels_array)
    squared_array = np.square(difference_array)
    mse = np.mean(squared_array)
    
    return mse


def RMSE_array(data_array, labels_array):
    """
    Root mean squared error with standard deviation between predicted and true sequence.

    Args:
        data (numpy array): predicted outputs
        labels (numpy array): ground truth outputs    
    """   
    difference_array = np.subtract(data_array, labels_array)
    squared_array = np.square(difference_array)
    mse = np.mean(squared_array)
    rmse = np.sqrt(mse)
    
    return rmse


def ME_array(data_array, labels_array):
    """
    Maximum error with standard deviation between predicted and true sequence.

    Args:
        data (numpy array): predicted outputs
        labels (numpy array): ground truth outputs    
    """   
    me = np.max(np.abs(data_array - labels_array))
    
    return me
