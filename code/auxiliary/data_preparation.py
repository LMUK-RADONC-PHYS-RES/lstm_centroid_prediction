"""
Created on June 2 2021

@author: Elia Lombardo

Functions for data loading
"""
# %%
import numpy as np
import torch
import os
import pandas as pd

# import self written modules
from auxiliary import utils

# %%


def sliding_wdws(data, wdw_size_i, wdw_size_o, step_size=1):
    """ Given a sequence of input data, subdivide it in input and output windows.
    Args:
        data: list with input data
        wdw_size_i: length of generated window to be used as input
        wdw_size_o: length of generated window to be used as output
        step_size: number of data points the window rolls at each step
    """
    x = []
    y = []

    # loop over the full sequence 
    for i in range(len(data) - wdw_size_i - wdw_size_o - 1):
        # select input and output windows 
        _x = data[step_size * i:(step_size * i + wdw_size_i)]
        _y = data[step_size * i + wdw_size_i:step_size * i + wdw_size_i + wdw_size_o]

        # keep the windows only if both input and output have expected size
        if len(_x) == wdw_size_i and len(_y) == wdw_size_o:
            x.append(_x)
            y.append(_y)

    return x, y


def sliding_wdws_vectorized(data, wdw_size_i=6, wdw_size_o=2, step_size=1):
    """ Given a sequence of input data, subdivide it in input and output windows (vectorized operations).
    Adapted from: https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5.
    Args:
        data: list with input data
        wdw_size_i: length of generated window to be used as input
        wdw_size_o: length of generated window to be used as output
        step_size: number of data points the window rolls at each step
    """
    start = 0
    stop = len(data)

    # find indices of all possible input windows using vectorized operations
    idx_windows_i = (start + 
        # create list with indices for first window, e.g. [0,1,2,3,4,5]
        np.expand_dims(np.arange(wdw_size_i), 0) +
        # create a column vector [0, step, 2*step, ...] to be added to first window list
        np.expand_dims(np.arange(stop - wdw_size_i - wdw_size_o + 1, step=step_size), 0).T)

    # find indices of output windows by taking for every row of the index_window_i matrix 
    # the last window_size_o nr of elements and add on top the window_size_o
    idx_windows_o = idx_windows_i[:, -wdw_size_o:] + wdw_size_o

    # print(idx_windows_i)
    # print(idx_windows_o)
    # print(data)
    # print(data[idx_windows_i]) # e.g. [[0.8,0.9,0.92,0.9],[0.8,0.74,0.42,0.44]]

    # return input and ouput data windows
    return data[idx_windows_i], data[idx_windows_o]


def get_wdws(data_snippets, wdw_size_i=6, wdw_size_o=2, step_size=1):
    """ Obtain a list with all sliding input and output windows from list with all data snippets. """
    input_wdw_list = []
    output_wdw_list = []

    for snippet in data_snippets:
        # print(snippet)
        input_wdws, output_wdws = sliding_wdws_vectorized(data=snippet, wdw_size_i=wdw_size_i, 
                                                          wdw_size_o=wdw_size_o, step_size=step_size)
        # print(output_wdws)

        # check if nr of input and output wdws is the same
        if len(input_wdws) != len(output_wdws):
            raise Exception("Attention! Nr of input and output windows unequal.")
        else:
            # append all windows for a given snippet in the list which will contain the windows for all snippets
            input_wdw_list.append(input_wdws)
            output_wdw_list.append(output_wdws)

    return input_wdw_list, output_wdw_list


# outlier detection and replacement
def replace_outliers(array, m_out=10.):
    """ Replace outliers in data window using the median of the distances from the median.
    Args:
        array: np.array with input data window
        m_out: multiple of the median of the distances from the median above which to replace the corrresponding value (outlier)
    """
    # compute distances from median
    d = np.abs(array - np.median(array))
    # print(d)

    # compute median of distances from median
    mdev = np.median(d)
    # print(mdev)

    # scale the distances with mdev
    s = d / (mdev if mdev else 1.)
    # print(s)
    
    # return data where outliers are replaced with median
    # print(np.where(s > m))
    array[np.where(s > m_out)] = np.median(array)

    return array 


def outlier_replacement_vectorized(array, seq_size=6, m_out=7):
    """ Given a sequence of data, subdivide it in windows (vecotrized operation) and then slide over them without overlapping to replace outliers without creating overlapping data.
    Adapted from: https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5.
    Args:
        array: input data
        seq_size: size of sliding window and at the same time step size during sliding
        m: multiple of the median of the distances from the median above which to replace the corrresponding value
    """
    start = 0
    stop = len(array)
    step_size = seq_size  # need to be the same to avoid replicating data

    # find indices of all possible windows using vectorized operations
    idx_windows = (start + 
        np.expand_dims(np.arange(seq_size), 0) +
        # Create a rightmost vector as [0, step, 2*step, ...].
        np.expand_dims(np.arange(stop - seq_size + 1, step=step_size), 0).T)

    # print(array[idx_windows]) # e.g. [[0.8,0.9,0.92,0.9],[0.8,0.74,0.42,0.44]]

    array_replaced = []
    # loop over all windows
    for seq in array[idx_windows]:
        # replace outliers for current window
        seq_replaced = replace_outliers(seq=seq, m_out=m_out)
        # extend current list by new (outlier replaced) window
        array_replaced.extend(seq_replaced)
    
    return array_replaced


def rolling_outlier_replacement(array, wdw_size=3, threshold_out=0.1):
    """ Replace outlieres based on deviation from median amplitude within sliding window using pandas rolling window.
    Adapted from: https://stackoverflow.com/questions/62692771/outlier-detection-based-on-the-moving-mean-in-python
    Args:
        array: list with input data as numpy array
        wdw_size: size of rolling wdw, with step size being automatically = 1 (pandas)     
        threshold_out: threshold amplitude above which to replace the corrresponding 
                        value (outlier) with the mdian within current window
    """
    # get normalized data to be used to find indices of data to be replaced -->
    # needed as threshold works best on all data if the amplitudes are comparable (i.e. normalized)
    array_norm = utils.normalize(array, {'actual': {'lower': np.min(array), 'upper': np.max(array)}, 
                                        'desired': {'lower': -1, 'upper': 1}})
    
    # put data into pandas dataframe
    df = pd.DataFrame({'data': array})
    df_norm = pd.DataFrame({'data': array_norm})

    # calculate rolling median for current element based on previous wdw_size nr of elements, 
    # else output NaN
    df['rolling_medians'] = df['data'].rolling(window=wdw_size, center=True).median()
    df_norm['rolling_medians'] = df_norm['data'].rolling(window=wdw_size, center=True).median()
    # print(f"df rolling medians: {df['rolling_medians']}")

    # calculate difference on normalized data
    df_norm['diff'] = df_norm['data'] - df_norm['rolling_medians']

    # find indices of values to be replaced with median value of window for normalized data
    diff = np.abs(df_norm['diff'].to_numpy())
    # print(diff)
    replace_flag = np.where(diff > threshold_out)
    # print(f'replace flag: {replace_flag}')

    replaced_array = np.copy(array)
    # return data where outliers are replaced with median of window
    replaced_array[replace_flag] = df['rolling_medians'].to_numpy()[replace_flag]

    return replaced_array  


def get_snippets(data, pauses_start, bhs, bhs_start, min_length_snippet=8):
    """ Separate  full data sequence into snippets according to image pauses and breathholds.
    Args:
        data: list with full input data sequence
        pauses_start: list with 1 for image pause start
        bhs: list with 1 for breath-holds
        bhs_start: list with 1 for breath-hold start
        min_length_snippet: int, minimum length of snippets which are not discarded
    """
    # 1. data separated according only to pauses (breath-holds included)
    data_snippets_with_bh = np.split(np.array(data), np.where(np.array(pauses_start) == 1)[0] + 1)

    # 2. data separated according to pauses and breath-holds start (breath-holds excluded)
    data_snippets_pause_bh = np.split(np.array(data), np.where((np.array(pauses_start) == 1) | (np.array(bhs_start) == 1))[0] + 1)
    # print(np.shape(data_snippets_pause_bh))

    # split also breath hold information according to image puases / bh start
    snippets_breathholds = np.split(np.array(bhs), np.where((np.array(pauses_start) == 1) | (np.array(bhs_start) == 1))[0] + 1)
    # print(np.shape(snippets_breathholds))

    # split image pauses snippets according to breath hold starts
    data_snippets_without_bh = []
    for idx, elem in enumerate(data_snippets_pause_bh):
        current_snippet = np.delete(elem, np.where(snippets_breathholds[idx] == 1))
        if len(current_snippet) > min_length_snippet:
            data_snippets_without_bh.append(current_snippet)

    return data_snippets_with_bh, data_snippets_without_bh


def load_data_snippets(path_data, train_cases=0, train_val_cases=0, 
                       direction='SI', curve_aspect='f_or', breathhold_inclusion=True,
                       normalization='case_based',
                       wdw_size_i=8, wdw_size_o=1, step_size=1,
                       save_min_max_amplitudes=False,
                       specific_patient=False,
                       phase='train'):
    """Load motion data snippets from xlsx file and select variant of data to be used.

    Args:
        path_data (str): path to folder with case subfolders or if specific_patient=True
                            directly to xlsx file containing COM positions.
        train_cases (int): absolute case number until which data counts as training set
        train_val_cases (int): absolute case nr until which dat counts as training+validation set
        direction (str, optional): select either Sup-Inf (SI) or Ant-Post (AP) COM motion. 
                                        Defaults to 'SI'.
        curve_aspect (str, optional): whether to take outlier replaced (or) or outlier
                                        replaced and moving average filtered COM curves.
                                        Defaults to 'or'.
        breathhold_inclusion (bool, optional): Whether to include breathholds in the data or not.
                                                Defaults to True.
        normalization (str, optional): 'case_based' applies a normalization based on the current
                                        motion curve's min and max.
                                        'population_based' applies a normalization based on the
                                        median min and median max of all cases
        wdw_size_i (int, optional): length of generated window to be used as input. Defaults to 8.
        wdw_size_o (int, optional): length of generated window to be used as output. Defaults to 1.
        step_size (int, optional): number of data points the window rolls at each step. Defaults to 1.
        save_min_max_amplitudes (bool, optional): whether to save excel file with min max amplitudes
                                                    used for normalization
        specific_patient: whether to get snippets only from a specific xlsx (i.e. video) file
        phase (str, optional): which data set to use. Defaults to 'train', also 'val' and 'test' possible.

    Raises:
        Exception: AP not implemented

    Returns:
        (x_list, y_list): lists containing arrays with sliding windows for different snippets
                            for the input (x) and output (y) of the models
    """
    
    # lists to contain input and output slinding windows from different snippets
    x_snippets = []
    y_snippets = []
    
    if save_min_max_amplitudes:
        # create list to conatin current file name, min and max amplitude
        ampl_info = [] 
        # initialize counter for videos found
        counter = 0
    
    if specific_patient is False:      
        if phase == 'train':
            list_path_cases = utils.subdir_paths(path_data)[:train_cases]
        if phase == 'val':
            list_path_cases = utils.subdir_paths(path_data)[train_cases:train_val_cases]
        if phase == 'test':
            list_path_cases = utils.subdir_paths(path_data)[train_val_cases:]
        
            
        print(f'\n\nLoading {phase} motion data...')
        # find excel files and get train data
        for path_case in list_path_cases:
            
            # eg 'case03'
            current_case = os.path.basename(path_case)
            print(f'\n-------- Current case: {current_case} -------\n')  

            # loop over all files of one case
            for _dir_name, _subdir_list, file_list in os.walk(path_case):
                for file_name in file_list:
                    
                    # check if file is xlsx
                    if file_name.endswith('.xlsx'):
                        x = [] 
                        y = [] 
                        
                        # load dataframe
                        print(f'Loading following file: {os.path.join(path_case, file_name)} ')
                        df = pd.read_excel(os.path.join(path_case, file_name), 
                                            engine='openpyxl')
                        
                        if direction == 'SI':
                            # get outlier replaced centroid posiiton data
                            if curve_aspect == 'or':
                                centroids = np.array(df['Target COM inf-sup (after outlier replacement) [mm]'].values)
                            # get outlier replaced + filtered centroid posiiton data
                            elif curve_aspect == 'f_or':
                                centroids = np.array(df['Target COM inf-sup (after smoothing) [mm]'].values)
                        elif direction == 'AP':
                            raise Exception('Attention: AP motion read-out to be implemented')
                        
                        # normalize data using video's min max
                        if normalization == 'case_based':
                            
                            if save_min_max_amplitudes:
                                # save current file name, min amplitued and max amplitude
                                ampl_info.append([os.path.join(path_case, file_name),
                                                np.min(centroids), np.max(centroids)]) 
                                counter += 1
                            
                            # perform normalization to range -1 +1 
                            centroids = utils.normalize(centroids, 
                                            {'actual': {'lower': np.min(centroids), 'upper': np.max(centroids)}, 
                                            'desired': {'lower': -1, 'upper': 1}})

                                                        
                        elif normalization == 'population based':
                            raise Exception('Attention: population based normalization not implemented')
                            
                        # separate data into snippets according to image pauses and bhs
                        try:
                            snippets_with_bh, \
                                snippets_without_bh = get_snippets(centroids, 
                                                                    pauses_start=df['Imaging paused'], 
                                                                    bhs=df['Breath-holds'], 
                                                                    bhs_start=df['Breath-holds start'])
                        # latest version of excel file contains 'Imaging pauses start'
                        except KeyError:
                            snippets_with_bh, \
                                snippets_without_bh = get_snippets(centroids, 
                                                                    pauses_start=df['Imaging paused start'], 
                                                                    bhs=df['Breath-holds'], 
                                                                    bhs_start=df['Breath-holds start'])
                                                    
                            
                            
                        # include bhs in motion curves
                        if breathhold_inclusion:
                            # get data input and ouput windows
                            x, y = get_wdws(snippets_with_bh, 
                                                    wdw_size_i=wdw_size_i, 
                                                    wdw_size_o=wdw_size_o, 
                                                    step_size=step_size)
                        # exclude bhs from motion curves
                        else:
                            # get data input and ouput windows
                            x, y = get_wdws(snippets_without_bh, 
                                                    wdw_size_i=wdw_size_i, 
                                                    wdw_size_o=wdw_size_o, 
                                                    step_size=step_size) 

                        #print(f'Number of snippets for current video: {np.shape(x)}')  # (2,)
                        
                        # extend list of snippet arrays with array containing
                        # windows of snippets of current file        
                        x_snippets.extend(x)    
                        y_snippets.extend(y) 
                        # np.shape(x_snippets[1])  # --> (nr_windows_from_current_snippet, wdw_size_i)                              
                        # np.shape(y_snippets[0])  # --> (nr_windows_from_current_snippet, wdw_size_o)  
        
        if save_min_max_amplitudes:
            # create excel list from data with min and max amplitudes
            column_names = ['File name', 'Min amplitude [mm]', 'Max amplitude [mm]']
            df = pd.DataFrame(ampl_info, columns=column_names)
            # save min max amplitudes to excel file
            df.to_excel(path_data + '/min_max_amplitudes_' + phase + \
                        '_BH_' + str(breathhold_inclusion) + '_' + \
                        curve_aspect + '_' + direction + '.xlsx', \
                            sheet_name='Peak amplitudes for normalization')
            
            
    if specific_patient:
        # load dataframe
        print(f'Loading following file: {path_data} ')
        df = pd.read_excel(path_data, engine='openpyxl')
        
        if direction == 'SI':
            # get outlier replaced centroid posiiton data
            if curve_aspect == 'or':
                centroids = np.array(df['Target COM inf-sup (after outlier replacement) [mm]'].values)
            # get outlier replaced + filtered centroid posiiton data
            elif curve_aspect == 'f_or':
                centroids = np.array(df['Target COM inf-sup (after smoothing) [mm]'].values)
        elif direction == 'AP':
            raise Exception('Attention: AP motion read-out to be implemented')
        
        # normalize data using video's min max
        if normalization == 'case_based':
            # perform normalization to range -1 +1 
            centroids = utils.normalize(centroids, 
                            {'actual': {'lower': np.min(centroids), 'upper': np.max(centroids)}, 
                            'desired': {'lower': -1, 'upper': 1}})

                                        
        elif normalization == 'population based':
            raise Exception('Attention: population based normalization to be implemented')
            
        # separate data into snippets according to image pauses and bhs
        try:
            snippets_with_bh, \
                snippets_without_bh = get_snippets(centroids, 
                                                    pauses_start=df['Imaging paused'], 
                                                    bhs=df['Breath-holds'], 
                                                    bhs_start=df['Breath-holds start'])
        # latest version of excel file contains 'Imaging pauses start'
        except KeyError:
            snippets_with_bh, \
                snippets_without_bh = get_snippets(centroids, 
                                                    pauses_start=df['Imaging paused start'], 
                                                    bhs=df['Breath-holds'], 
                                                    bhs_start=df['Breath-holds start'])

        # include bhs in motion curves
        if breathhold_inclusion:
            # get data input and ouput windows
            x, y = get_wdws(snippets_with_bh, 
                                    wdw_size_i=wdw_size_i, 
                                    wdw_size_o=wdw_size_o, 
                                    step_size=step_size)
        # exclude bhs from motion curves
        else:
            # get data input and ouput windows
            x, y = get_wdws(snippets_without_bh, 
                                    wdw_size_i=wdw_size_i, 
                                    wdw_size_o=wdw_size_o, 
                                    step_size=step_size) 

        # print(f'Number of snippets for current video: {np.shape(x)}')  # (2,)
        
        # extend list of snippet arrays with array containing
        # windows of snippets of current file        
        x_snippets.extend(x)    
        y_snippets.extend(y) 
        # np.shape(x_snippets[1])  # --> (nr_windows_from_current_snippet, wdw_size_i)                              
        # np.shape(y_snippets[0])  # --> (nr_windows_from_current_snippet, wdw_size_o)  
                               
    return x_snippets, y_snippets


def get_data_batches(data, batch_size, concat=True, 
                     to_tensor=True, gpu_usage=False, device=None):
    """Get batches of data from data snippets.

    Args:
        data (list): list with data snippets with shape=(nr_snippets,) 
                        or windows with shape=(nr_wdws, wdw_size)
        batch_size (int): number of data windows in one batch
        concat (bool, optional): if True, concatenate all snippets in the data list
                                    to an array with shape=(nr_wdws, wdw_size) . Defaults to True.
        to_tensor (bool, optional): convert data to Pytorch tensor. Defaults to True.
        gpu_usage (bool, optional): transfer data to GPU. Defaults to False.
        device (Pytorch device, optional): GPU device to be used. Defaults to None.

    Returns:
        (array or tensor): data batches with shape (nr_batches, batch_size, wdw_size_i)
    """
    
    if concat:
        # concatanate wdws from different snippets
        # N.B. concatenation automatically drops empty arrays
        data = np.concatenate(data, axis=0)    
        
    batches = []  
    # loop to subdivide data into batches 
    for idx in range(len(data) // batch_size):
        current_batch = data[idx * batch_size: idx * batch_size + batch_size]
        batches.append(current_batch) 
    
    if to_tensor:
        batches = torch.tensor(batches, dtype=torch.float32)
        if gpu_usage:
            batches = batches.to(device)
    else:
        batches = np.array(batches)

    
    return batches


def get_dummy_data_xy_batches(size, wdw_size_i=5, wdw_size_o=1, 
                           step_size=1, batch_size=128, 
                           gpu_usage=False, device=None):
    """Generate dummy data to perform sanity checks on network memory. The idea is 
    to have 1s at certain intervals to see if networks with different
    memory lengths are able to predict them.

    Args:
        size (int): length of dummy time series
        wdw_size_i (int, optional): length of generated window to be used as input
        wdw_size_o (int, optional): length of generated window to be used as output
        step_size (int, optional): number of data points the window rolls at each step
        batch_size (int, optional): number of windows to be fed to net. Defaults to 128.
        gpu_usage (bool, optional): transfer data to GPU. Defaults to False.
        device (Pytorch device, optional): GPU device to be used. Defaults to None.

    Returns:
        (array or tensor): data batches with shape (nr_batches, batch_size, wdw_size_i)
    """
    
    data = np.zeros(shape=(size))
    
    # set a 1 every 5 points so that network has to learn that
    # after 5 steps a 1 is followed by another 1
    for idx in range(len(data) // 5):
        data[5 * idx] = 1
    
    # divide data in input and output wdws
    x_data, y_data = sliding_wdws_vectorized(data=data, 
                                            wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o, 
                                            step_size=step_size)
    # print(x_data)   # [[1. 0. 0. 0. 0.] ...
    
    # get batches of data
    x_data_batches = get_data_batches(data=x_data, batch_size=batch_size, 
                                                    concat=False,
                                                    to_tensor=True, 
                                                    gpu_usage=gpu_usage, device=device)

    y_data_batches = get_data_batches(data=y_data, batch_size=batch_size, 
                                                    concat=False,
                                                    to_tensor=True, 
                                                    gpu_usage=gpu_usage, device=device)
        
    return x_data_batches, y_data_batches

# %%


# run load_data_snippets function ot get excel with amplitudes for normalization
# print('Getting amplitudes for normalization...')
# x, y = load_data_snippets(path_data=os.path.join('/home/centroid_prediction/data', 
#                             'preprocessed', 
#                             '2021_06_16_respiratory_patients_LMU_ogv-no_norm'), 
#                             curve_aspect='f_or', breathhold_inclusion=True,
#                             save_min_max_amplitudes=True,
#                             phase='train')

