import numpy as np
import time
import torch
import torch.nn as nn
import os
import joblib
from torch.utils.tensorboard import SummaryWriter

# import self written modules
from auxiliary import metrics, utils, plotting


def train_model(model, train_data, train_labels, loss_name, optimizer,
                epochs, early_stopping=None,
                val_data=None, val_labels=None, path_saving=None, path_tb=None):
    """Function to train a Pytorch network and score some metrics.

    Args:
        model (Pytorch model): Network architecture
        train_data (Pytorch tensors): array of shape (nr_batches, batch_size, wdw_size_i) with 
                                        input data subdiveded in batches
        train_labels (Pytorch tensors): array of shape (nr_batches, batch_size, wdw_size_i) with 
                                        output data subdiveded in batches
        loss_name (str): name of loss function, eg 'MSE'
        optimizer (Pytorch optimizer): optimizer used to update network's weights
        epochs (int): total number of training epochs
        early_stopping (int, optional): whether to stop the optimization if the loss does
                                not get better after early_stopping nr of epochs. Defaults to None.
        val_data (Pytorch tensors, optional): array with validation input data. Defaults to None.
        val_labels (Pytorch tensors, optional): array with validation output data. Defaults to None.
        path_saving (str): path to folder where models are saved. Defaults to None.
        path_tb (str): path to folder where tensorboard runs are saved. 
                Must match tensorboard -- logdir path. 
                Defaults to None in which case no writer is initialized.

    Returns:
        train_losses (list): training losses for different epochs
        val_losses (list): val losses for different epochs
        y_train_pred (Pytorch tensor): predicted output windows for last training batch, 
                        size = [batch_size, wdw_size_o] 
        y_val_pred (Pytorch tensor): predicted output windows for last validation batch, 
                        size = [batch_size, wdw_size_o] 
        tot_t (float): total time needed for optimization 
    """

    # set writer for tensorboard monitoring
    if path_tb is not None:
        writer = SummaryWriter('/home/centroid_prediction/results/tensorboard_runs')
     
    if loss_name == 'MSE':
        loss_function = nn.MSELoss()
    
    train_losses = [] 
    val_losses = [] 
    val_losses_poi = []  # loss for the point of interest, i.e. last time point
    best_val_loss = 1000000  # to be sure loss decreases

    t0 = time.time()
    
    # loop over all epochs
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        
        model.train()
        epoch_loss_train = [] 
        # loop over all batches of data
        for x_train_batch, y_train_batch in zip(train_data, train_labels):
            # print(f'Shape of x_batch: {x_train_batch.shape}')  # (batch_size, wdw_size_i)
            # print(f'Shape of y_batch: {y_train_batch.shape}')  # (batch_size, wdw_size_o)
            
            # clear stored gradients
            optimizer.zero_grad()

            # forward pass
            y_train_pred = model(x_train_batch)

            # compute the loss for current batch
            loss = loss_function(y_train_pred.float(), y_train_batch)
            epoch_loss_train.append(loss.item())

            # back propagate the errors and update the weights within batch
            loss.backward()
            optimizer.step()
            
        # compute loss for current epoch by averaging over batch losses and store results
        epoch_loss_train = np.mean(epoch_loss_train) 
        train_losses.append(epoch_loss_train)
        
        if path_tb is not None:
            writer.add_scalar(f"train_{loss_name}_loss", epoch_loss_train, epoch)
           
        if val_data is not None:
            model.eval()
            epoch_loss_val = [] 
            epoch_loss_val_poi = [] 
            
            with torch.no_grad():
                for x_val_batch, y_val_batch in zip(val_data, val_labels):
                    
                    y_val_pred = model(x_val_batch)

                    loss_val = loss_function(y_val_pred.float(), y_val_batch)
                    loss_val_poi = loss_function(y_val_pred[:, -1].float(), y_val_batch[:, -1])
                    epoch_loss_val.append(loss_val.item())
                    epoch_loss_val_poi.append(loss_val_poi.item())
                    
            # compute loss for current epoch by averaging over batch losses
            epoch_loss_val = np.mean(epoch_loss_val)
            epoch_loss_val_poi = np.mean(epoch_loss_val_poi)
            val_losses.append(epoch_loss_val) 
            val_losses_poi.append(epoch_loss_val_poi) 
 
            # save data for tensorboard
            if path_tb is not None:
                writer.add_scalar(f"val_{loss_name}_loss", epoch_loss_val, epoch)
                                   
            # save model if validation loss improves
            if val_losses[-1] < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_losses[-1]
                best_val_loss_poi = val_losses_poi[-1]
                if path_saving is not None:
                    torch.save(model.state_dict(), os.path.join(
                        path_saving, f'best_model_epoch_{best_epoch}_{loss_name}_' + \
                        f'val_loss_poi_{best_val_loss_poi:.6f}.pth'))
                    print('...saved model based on new best validation loss') 
                    
            if epoch % 1 == 0:
                print(f'Train loss: {epoch_loss_train} - '
                      f'Val loss: {epoch_loss_val} - '
                      f'Best val loss: {best_val_loss} - '      
                      f'Best val loss poi: {best_val_loss_poi} ') 
       
            # stop the optimization if the loss didn't decrease after early_stopping nr of epochs
            if early_stopping is not None:
                if (epoch - best_epoch) > early_stopping:
                    print('Early stopping the optimization!')
                    break
                                        
        else:
            if epoch % 1 == 0:
                print(f'Train loss: {epoch_loss_train}')
    
    # close tensorboard writer 
    if path_tb is not None:          
        writer.close()
      
    t1 = time.time()
    tot_t = round((t1 - t0) / 60)
    print(f'\n------------ Total time needed for optimization: {tot_t} min --------- ') 
    
    return train_losses, val_losses, val_losses_poi, y_train_pred, y_val_pred, tot_t


def train_closed_LR(model, train_data, train_labels, 
                    val_data=None, val_labels=None, path_saving=None):
    """Function to train and validate closed form linear regression.

    Args:
        model (sklearn model): initiliazed Ridge class
        train_data (np.array): array of shape (nr_batches, 1, wdw_size_i) with 
                                        input data windows subdiveded in batches
        train_labels (np.array): array of shape (nr_batches, 1, wdw_size_i) with 
                                        output data windows subdiveded in batches
        val_data (np.array): array of shape (nr_batches, 1, wdw_size_i) with 
                                        input validation data windows subdiveded in batches. Defaults to None.
        val_labels (np.array): array of shape (nr_batches, 1, wdw_size_i) with 
                                        output validation data windows subdiveded in batches. Defaults to None.

    Returns:
        train_loss: final training evaluation metric
        val_loss: final validation evaluation metric
        val_loss_poi: final validation evaluation metric for prediction point of interest
        y_train_pred: predicted output windows for training data (full batch), 
                        size = [batch_size, wdw_size_o] 
        y_val_pred: predicted output windows for last validation data (full batch), 
                        size = [batch_size, wdw_size_o] 
        tot_t (float): total time needed for optimization 
    """
    
    # change shape of train&val batches to (nr_batches, wdw_size)
    train_data_full_batch = train_data[:, 0, :] 
    train_labels_full_batch = train_labels[:, 0, :] 
    val_data_full_batch = val_data[:, 0, :] 
    val_labels_full_batch = val_labels[:, 0, :] 
    # print(train_data_full_batch.shape)  # (35106, 16)
    # print(train_labels_full_batch.shape)  # (36106, 3)
    
    print('Starting optimization...')
    t0 = time.time()
    # fit model on training data set
    model.fit(train_data_full_batch, train_labels_full_batch)
    t1 = time.time()
    tot_t = round((t1 - t0) * 1000)
    print(f'---------- Total time needed for optimization: {tot_t} ms ------- ') 

    
    # validate model on train set
    y_train_pred = model.predict(train_data_full_batch)
    train_loss = metrics.MSE_array(y_train_pred, train_labels_full_batch)
    
    # validate model on validation set
    y_val_pred = model.predict(val_data_full_batch)
    val_loss = metrics.MSE_array(y_val_pred, val_labels_full_batch)
    val_loss_poi = metrics.MSE_array(y_val_pred[:, -1], val_labels_full_batch[:, -1]) 
    
    if path_saving is not None:
        joblib.dump(model, os.path.join(path_saving, 'trained_model.pkl'))  
       
    return train_loss, val_loss, val_loss_poi, y_train_pred, y_val_pred, tot_t



def train_val_model_online(model, train_val_data, train_val_labels, 
                        loss_name, optimizer,
                        wdw_size_i, wdw_size_o,
                        min_train_data_length=80,
                        online_epochs=100,
                        output_positions=False,
                        plot_path_saving=None,
                        info='val'):
    """Function to train a Pytorch network on first window of data and then train and
    validate it continuosly (i.e. online) on the remaining sliding windows. 
    By setting online_epochs to None it is possible to validate only.

    Args:
        model (Pytorch model): Network architecture
        train_val_data (Pytorch tensors): array of shape = (nr_batches, 1, wdw_size_i) with 
                                        input data (for current video) subdivided in batches of size=1. 
        train_val_labels (Pytorch tensors): array of shape = (nr_batches, 1, wdw_size_o) with 
                                        output data (for current video) subdivided in batches of size=1. 
        loss_name (str): name of loss function, eg 'MSE'
        optimizer (Pytorch optimizer): optimizer used to update network's weights
        wdw_size_i (int): length of single input data window.
        wdw_size_o (int): lookahead window for prediction. For instance 1=250 ms, 2=500ms and 3=750ms
        min_train_data_length (int): nr of data point to be contained in set of input training windows 
        online_epochs (int): total number of online sliding window training epochs. 
                                If None, no online training is performed.
        output_positions (bool): if True, accumulate and return predicted and ground truth centroid positions
                                for selected lookahead time.
        info (str): some string for predicted window plot name

    Returns:
        train_online_epoch_losses_wdws (list of lists): losses as a function of epochs and windows for online training
        train_online_losses_wdws(list): online training losses as a function of windows 
        val_losses_wdws (list): validation losses as a function of windows 
        val_losses_poi_wdws (list): validation losses as a function of windows for point of interest
        val_loss_video (float): average loss over all windows for current video
        val_loss_poi_video (float): average loss over all windows for point of interest for current video
        y_val_pred (list): predicted data for last input window 
        tot_t_online (int):  time needed to complete online training    
    """
     
    if loss_name == 'MSE':
        loss_function = nn.MSELoss()
    
    train_online_losses_wdws = [] 
    val_losses_wdws = [] 
    val_losses_poi_wdws = []  # loss for the point of interest, i.e. last time point
    tot_t_online = 0
    
    # whether to accumulate ground truth and predicted amplitudes
    if output_positions:
        y_batch_video = []
        y_pred_video = []   
 
    # loop over all windows of data, one batch containing one window (batch_size=1)
    for wdw_nr in range(len(train_val_data) - (min_train_data_length - wdw_size_i + 1) - wdw_size_o):
        # get indices for set of windows with length 
        # min_train_data_length for online training
        wdw_nr_train_start = wdw_nr 
        wdw_nr_train_stop = wdw_nr + (min_train_data_length - wdw_size_i) 
        # get index of currently available validation input window
        # and of ground truth output window (of course not possible in real-time scenario,
        # only used to compute validation metric) 
        wdw_nr_val = wdw_nr_train_stop + wdw_size_o
        
        # get validation input and output window
        x_val_batch, y_val_batch = train_val_data[wdw_nr_val], train_val_labels[wdw_nr_val]
        # print(f'x_val_batch.shape: {x_val_batch.shape}')  # torch.size([1, wdw_size_i])
        # print(f'y_val_batch.shape: {y_val_batch.shape}')  # torch.size([1, wdw_size_o])
                
        # set model to eval model for validation  
        model.eval()

        # forward pass
        t0_forward = time.time()
        y_val_pred = model(x_val_batch) 
        t1_forward = time.time()
        print(f'Time need for forward pass: {round((t1_forward - t0_forward) * 1000)} ms')
        # print(y_val_pred.shape)  # torch.Size([1, 3])
        # print(f'...sending prediction to MLC for wdw_nr {wdw_nr}')

        if output_positions:
            # store predicted and ground truth centroid position for current video
            # for later plotting
            y_batch_video.append(y_val_batch[:, -1])
            y_pred_video.append(y_val_pred[:, -1].float())
         
        # plot input and ground truth vs predicted output wdw for validation   
        if plot_path_saving is not None:
            plotting.predicted_wdw_plot(x=x_val_batch, y=y_val_batch, 
                                y_pred=y_val_pred, wdw_nr=wdw_nr_val, 
                                last_pred=False,
                                display=True, save=True, 
                                path_saving=plot_path_saving, 
                                info=info)
        
        # compute val loss, this of course would not be possible in real-time clinical practice
        loss_val = loss_function(y_val_pred.float(), y_val_batch)
        loss_val_poi = loss_function(y_val_pred[:, -1].float(), y_val_batch[:, -1]) 

        # append loss for current window
        val_losses_wdws.append(loss_val.item())  
        val_losses_poi_wdws.append(loss_val_poi.item())  

        # training (happens 'after' validation for iterative model as
        # optimization takes some time)
        if online_epochs is not None:
            # get input and output set of windows for online training
            x_train_batch = train_val_data[wdw_nr_train_start:wdw_nr_train_stop + 1, 0, :] 
            y_train_batch = train_val_labels[wdw_nr_train_start:wdw_nr_train_stop + 1, 0, :]
            # print(f'x_train_batch.shape: {x_train_batch.shape}') # torch.size([nr wdws to reach min_train_data_length, wdw_size_i])
            # print(f'y_train_batch.shape: {y_train_batch.shape}')  # torch.size([nr wdws to reach min_train_data_length, wdw_size_o]) 
                    
            train_losses_online = []      
            val_losses_online = []      
            t0_online = time.time()
            # print('Online training...')
            # loop over all epochs of online training
            for epoch in range(online_epochs):
                model.train()
            
                # clear stored gradients
                optimizer.zero_grad()

                # forward pass
                y_online_train_pred = model(x_train_batch)

                # compute the loss for current windows
                loss_train_online = loss_function(y_online_train_pred.float(), y_train_batch)
                           
                # backpropagate the errors and update the weights for current window
                loss_train_online.backward()
                optimizer.step()
                
                if plot_path_saving is not None:
                    # append losses from online training over different epochs
                    train_losses_online.append(loss_train_online.item())
                    # compute val loss for plotting
                    loss_val_online = loss_function(model(x_val_batch).float(), y_val_batch)
                    val_losses_online.append(loss_val_online.item())

            # append final loss for current window
            train_online_losses_wdws.append(loss_train_online)
                    
            t1_online = time.time()    
            tot_t_online = round((t1_online - t0_online) * 1000)
            print(f'---- Total time needed for online training: {tot_t_online} ms -----\n')  
     
            if plot_path_saving is not None:
                # plot input and output ground truth vs predicted wdw  
                # for the last element in the set of online training wdws
                plotting.predicted_wdw_plot(x=x_train_batch, y=y_train_batch, 
                                y_pred=y_online_train_pred, wdw_nr=wdw_nr_train_stop, 
                                last_pred=False,
                                display=True, save=True, 
                                path_saving=plot_path_saving, 
                                info='train')
                
                # plot online training and validation losses as a function of epochs
                plotting.losses_plot_detailed(train_losses=train_losses_online,
                                loss_fn=loss_name, display=False, save=True, 
                                path_saving=plot_path_saving, 
                                info_loss='train_online_wdw_nr_' + str(wdw_nr_train_stop) + '_') 
                plotting.losses_plot_detailed(val_losses=val_losses_online,
                                loss_fn=loss_name, display=False, save=True, 
                                path_saving=plot_path_saving, 
                                info_loss='val_online_wdw_nr_' + str(wdw_nr_train_stop) + '_')            
            
    # average losses of all windows (but the first one) to build validation loss
    # for current video       
    val_loss_video = np.mean(val_losses_wdws)               
    val_loss_poi_video = np.mean(val_losses_poi_wdws)
    
        
    if output_positions:
        return train_online_losses_wdws, \
                val_losses_wdws, val_losses_poi_wdws, \
                val_loss_video, val_loss_poi_video, \
                y_pred_video, y_batch_video, \
                tot_t_online           
    else:    
        return train_online_losses_wdws, \
                val_losses_wdws, val_losses_poi_wdws, \
                val_loss_video, val_loss_poi_video, \
                tot_t_online    

 
def train_val_closed_LR_online(model, train_val_data, train_val_labels, 
                        wdw_size_i, wdw_size_o,
                        min_train_data_length=120,
                        offline_training=False,
                        output_positions=False,
                        plot_path_saving=None,
                        info='val'):
    """Function to train an validate sklearn model continuosly (i.e. online) on sliding windows.

    Args:
        model (sklearn model): initiliazed Ridge class
        train_val_data (np.array): array of shape = (nr_batches, 1, wdw_size_i) with 
                                        input data (for current video) subdivided in batches of size=1. 
        train_val_labels (np.array): array of shape = (nr_batches, 1, wdw_size_o) with 
                                        output data (for current video) subdivided in batches of size=1. 
        wdw_size_i (int): length of single input data window.
        wdw_size_o (int): lookahead window for prediction. For instance 1=250 ms, 2=500ms and 3=750ms
        min_train_data_length (int): nr of data point to be contained in set of input training windows 
        offline_training (bool): if True, online training is performed, else only validation is done
        output_positions (bool): if True, accumulate and return predicted and ground truth centroid positions
                                for selected lookahead time.
        plot_path_saving (str): If not None, path where plots of predicted wdws are saved.
        info (str): some string for predicted window plot name

    Returns:
        train_online_losses_wdws(list): online training losses as a function of windows 
        val_losses_wdws (list): validation losses as a function of windows 
        val_losses_poi_wdws (list): validation losses as a function of windows for point of interest
        val_loss_video (float): average loss over all windows for current video
        val_loss_poi_video (float): average loss over all windows for point of interest for current video
        y_val_pred (list): predicted data for last input window 
        tot_t_online (int):  time needed to complete online training    
    """
    
    train_online_losses_wdws = [] 
    val_losses_wdws = [] 
    val_losses_poi_wdws = []  # loss for the point of interest, i.e. last time point
    tot_t_online = 0
    
    # whether to accumulate ground truth and predicted amplitudes
    if output_positions:
        y_batch_video = []
        y_pred_video = []   
 
    # loop over all windows of data, one batch containing one window (batch_size=1)
    for wdw_nr in range(len(train_val_data) - (min_train_data_length - wdw_size_i + 1) - wdw_size_o):
        
        # get indices for set of windows with length 
        # min_train_data_length for online training
        wdw_nr_train_start = wdw_nr 
        wdw_nr_train_stop = wdw_nr + (min_train_data_length - wdw_size_i) 
        # get index of currently available validation input window
        # and of ground truth output window (of course not possible in real-time scenario,
        # only used to compute validation metric) 
        wdw_nr_val = wdw_nr_train_stop + wdw_size_o
  
        # training     
        if offline_training is False:
            # get set of input and output windows 
            x_train_batch = train_val_data[wdw_nr_train_start:wdw_nr_train_stop + 1, 0, :]
            y_train_batch = train_val_labels[wdw_nr_train_start:wdw_nr_train_stop + 1, 0, :]
            # print(f'x_train_batch.shape: {x_train_batch.shape}')  # (nr wdws to reach min_train_data_length, wdw_size_i)
            # print(f'y_train_batch.shape: {y_train_batch.shape}')  # (nr wdws to reach min_train_data_length, wdw_size_o)
                
                    
            t0_online = time.time()
            # print('Online training...')
            
            model.fit(x_train_batch, y_train_batch)

            # forward pass to check performance
            y_online_train_pred = model.predict(x_train_batch)

            # compute the loss for current window
            loss_train_online = metrics.MSE_array(y_online_train_pred, y_train_batch)

            # append final loss for current window
            train_online_losses_wdws.append(loss_train_online)
                    
            t1_online = time.time()    
            tot_t_online = round((t1_online - t0_online) * 1000)
            # print(f'------ Total time needed for online training: {tot_t_online} ms -------\n')  

            # plot input and ground truth vs predicted output wdw for online training   
            if plot_path_saving is not None:
                plotting.predicted_wdw_plot(x=x_train_batch, y=y_train_batch, 
                                y_pred=y_online_train_pred, wdw_nr=wdw_nr_train_stop, 
                                last_pred=False,
                                display=True, save=True, 
                                path_saving=plot_path_saving, 
                                info='train')
            

        # validation
        
        # get input and output wdw 
        x_val_batch = train_val_data[wdw_nr_val]
        y_val_batch = train_val_labels[wdw_nr_val]
        # print(f'x_val_batch.shape: {x_val_batch.shape}')  # (1, wdw_size_i)
        # print(f'y_val_batch.shape: {y_val_batch.shape}')  # (1, wdw_size_o)   
             
        # forward pass
        y_val_pred = model.predict(x_val_batch)
        # print(f'...sending prediction to MLC for wdw_nr {wdw_nr}')

        if output_positions:
            # store predicted and ground truth centroid position for current video
            # for later plotting
            y_batch_video.append(y_val_batch[:, -1])
            y_pred_video.append(y_val_pred[:, -1])
        
        # plot input and ground truth vs predicted output wdw for validation   
        if plot_path_saving is not None:
            plotting.predicted_wdw_plot(x=x_val_batch, y=y_val_batch, 
                                y_pred=y_val_pred, wdw_nr=wdw_nr_val, 
                                last_pred=False,
                                display=True, save=True, 
                                path_saving=plot_path_saving, 
                                info=info)
        
        # compute val loss, this of course would not be possible in real-time clinical practice
        loss_val = metrics.MSE_array(y_val_pred, y_val_batch)
        loss_val_poi = metrics.MSE_array(y_val_pred[:, -1], y_val_batch[:, -1]) 

        # append loss for current window
        val_losses_wdws.append(loss_val.item())  
        val_losses_poi_wdws.append(loss_val_poi.item()) 
 
    # average losses of all windows (but the first one) to build validation loss
    # for current video       
    val_loss_video = np.mean(val_losses_wdws)               
    val_loss_poi_video = np.mean(val_losses_poi_wdws)
    
        
    if output_positions:
        return train_online_losses_wdws, \
                val_losses_wdws, val_losses_poi_wdws, \
                val_loss_video, val_loss_poi_video, \
                y_pred_video, y_batch_video, \
                tot_t_online           
    else:    
        return train_online_losses_wdws, \
                val_losses_wdws, val_losses_poi_wdws, \
                val_loss_video, val_loss_poi_video, \
                tot_t_online    
   
              
