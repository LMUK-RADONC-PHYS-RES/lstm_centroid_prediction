"""
Created on November 2021

@author: Elia Lombardo

Main script to test the statistical significance of differences between models
"""

# %%
import config
from auxiliary import plotting

# %%

import scipy.stats
import scikit_posthocs 
import os
import numpy as np

# %%
# set paths to RMSE obtained with different models

cohort = 'LMU_with_BH'   # 'LMU_without_BH', 'Gemelli', 'LMU_with_BH'
forecast = 750   # 250, 500, 750



# offline LSTM
if cohort == 'LMU_without_BH':
    if forecast == 250:
        path_curves_1 = config.path_results + '/LSTM_stateless/test/2021-10-05-12:51:58'  # 250 ms
    if forecast == 500: 
        path_curves_1 = config.path_results + '/LSTM_stateless/test/2021-10-05-12:29:55'  # 500 ms
    if forecast == 750:
        path_curves_1 = config.path_results + '/LSTM_stateless/test/2021-10-05-09:58:52'  # 750 ms
if cohort == 'Gemelli':
    if forecast == 250:    
        path_curves_1 = config.path_results + '/LSTM_stateless/test/2021-10-28-13:23:38'  # 250 ms
    if forecast == 500:
        path_curves_1 = config.path_results + '/LSTM_stateless/test/2021-10-28-13:25:32'  # 500 ms
    if forecast == 750:
        path_curves_1 = config.path_results + '/LSTM_stateless/test/2021-10-28-13:26:33'  # 750 ms
if cohort == 'LMU_with_BH':
    if forecast == 250:
        path_curves_1 = config.path_results + '/LSTM_stateless/test/2021-10-27-09:54:18'  # 250 ms
    if forecast == 500:
        path_curves_1 = config.path_results + '/LSTM_stateless/test/2021-10-27-09:55:41'  # 500 ms
    if forecast == 750:
        path_curves_1 = config.path_results + '/LSTM_stateless/test/2021-10-27-09:58:21'  # 750 ms

# offline+online LSTM
if cohort == 'LMU_without_BH':
    if forecast == 250:
        path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-20-07:09:00_online'  # 250 ms
    if forecast == 500: 
        path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-11-07:54:05_online'  # 500 ms
    if forecast == 750:
        path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-20-07:05:49_online'  # 750 ms
if cohort == 'Gemelli':
    if forecast == 250:    
        path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-28-13:31:17_online'
    if forecast == 500:
        path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-28-13:32:12_online'
    if forecast == 750:
        path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-28-13:33:02_online'
if cohort == 'LMU_with_BH':
    if forecast == 250:
        path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-27-12:30:01_online'
    if forecast == 500:
        path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-27-12:31:24_online'
    if forecast == 750:
        path_curves_2 = config.path_results + '/LSTM_stateless/test/2021-10-27-12:32:16_online'



# offline LR
if cohort == 'LMU_without_BH':
    if forecast == 250:
        path_curves_3 = config.path_results + '/LR_closed_form/test/2021-10-20-07:18:03'   # 250 ms
    if forecast == 500: 
        path_curves_3 = config.path_results + '/LR_closed_form/test/2021-10-11-08:00:47'   # 500 ms
    if forecast == 750:
        path_curves_3 = config.path_results + '/LR_closed_form/test/2021-10-20-07:15:20'   # 750 ms
if cohort == 'Gemelli':
    if forecast == 250:    
        path_curves_3 = config.path_results + '/LR_closed_form/test/2021-10-28-13:01:49'
    if forecast == 500:
        path_curves_3 = config.path_results + '/LR_closed_form/test/2021-10-28-13:04:35'
    if forecast == 750:
        path_curves_3 = config.path_results + '/LR_closed_form/test/2021-10-28-13:05:19'
if cohort == 'LMU_with_BH':
    if forecast == 250:
        path_curves_3 = config.path_results + '/LR_closed_form/test/2021-10-27-08:57:34'
    if forecast == 500:
        path_curves_3 = config.path_results + '/LR_closed_form/test/2021-10-27-09:17:39'
    if forecast == 750:
        path_curves_3 = config.path_results + '/LR_closed_form/test/2021-10-27-09:19:55'



# online LR
if cohort == 'LMU_without_BH':
    if forecast == 250:
        path_curves_4 = config.path_results + '/LR_closed_form/test/2021-10-06-08:54:00_online'   # 250 ms
    if forecast == 500: 
        path_curves_4 = config.path_results + '/LR_closed_form/test/2021-10-06-08:52:39_online'  # 500 ms
    if forecast == 750:
        path_curves_4 = config.path_results + '/LR_closed_form/test/2021-10-06-08:37:11_online'  # 750 ms
if cohort == 'Gemelli':
    if forecast == 250:    
        path_curves_4 = config.path_results + '/LR_closed_form/test/2021-10-28-13:19:14_online'
    if forecast == 500:
        path_curves_4 = config.path_results + '/LR_closed_form/test/2021-10-28-13:19:57_online'
    if forecast == 750:
        path_curves_4 = config.path_results + '/LR_closed_form/test/2021-10-28-13:20:30_online'
if cohort == 'LMU_with_BH':
    if forecast == 250:
        path_curves_4 = config.path_results + '/LR_closed_form/test/2021-10-27-09:43:05_online'
    if forecast == 500:
        path_curves_4 = config.path_results + '/LR_closed_form/test/2021-10-27-09:43:53_online'
    if forecast == 750:
        path_curves_4 = config.path_results + '/LR_closed_form/test/2021-10-27-09:46:39_online'


# load testing RMSE model 1,2,3,4
rmses_1 = np.loadtxt(os.path.join(path_curves_1, 'rmses_mm.txt'))
rmses_2 = np.loadtxt(os.path.join(path_curves_2, 'rmses_mm.txt'))
rmses_3 = np.loadtxt(os.path.join(path_curves_3, 'rmses_mm.txt'))
rmses_4 = np.loadtxt(os.path.join(path_curves_4, 'rmses_mm.txt'))

# as online LR had different input_wdw_size, some slicing needs to be done to 
# exclude windows which were not seen by other models
if cohort == 'LMU_without_BH':
    if forecast == 250:
        rmses_4 = np.delete(rmses_4, [57, 15, 13, 6, 54, 24, 47, 52, 39, 55])
    if forecast == 500:
        rmses_4 = np.delete(rmses_4, [60, 17, 15, 8, 57, 37, 26, 50, 55, 42, 58, 59, 20, 4, 21])
    if forecast == 750:
        rmses_4 = np.delete(rmses_4, [60, 17, 15, 8, 57, 37, 26, 50, 55, 42, 58, 59, 20, 4, 21, 2])  
if cohort == 'LMU_with_BH':
    if forecast == 250:
        rmses_4 = np.delete(rmses_4, [36])
    if forecast == 500:
        rmses_4 = np.delete(rmses_4, [36, 51])
    if forecast == 750:
        rmses_4 = np.delete(rmses_4, [36, 51])
# %%
# perform Friedman test to check if there is any significant difference
# between the RMSE obtained with the different models (measurements) and if
# there is one perform post hoc test to see which groups are pairwise significantly differnt

# txt file to save prints to
fn = f'stats_{cohort}_forecast_{forecast}.txt' 
with open(os.path.join(config.path_results, 'various/statistical_tests', fn), 'w') as f:
    # perform Friedman Test
    statistic, friedman_pvalue = scipy.stats.friedmanchisquare(rmses_1, rmses_2, rmses_3, rmses_4)
    print(f'Friedman statistic and pvalue: \n {statistic} and {friedman_pvalue}', file=f)

    if friedman_pvalue < 0.05:
        print('Friedmann test was significant!')
        # perform post hoc test if friedman test is significant
        data = np.array([rmses_1, rmses_2, rmses_3, rmses_4])
        # transpose data matrix as the columns in the posthoc test 
        # are the groups( i.e. models) by default
        posthoc_pvalues = scikit_posthocs.posthoc_nemenyi_friedman(data.T)
        print(f'\nPosthoc pvalue matrix:\n {posthoc_pvalues}', file=f)
            
        print('\nStatistically different means are found between groups:', file=f)
        for model_nr_a, model_nr_b in zip(np.where(posthoc_pvalues < 0.05)[0], np.where(posthoc_pvalues < 0.05)[1]):
            print(f'Model nr {model_nr_a + 1} and model nr {model_nr_b + 1} p-value: {posthoc_pvalues[model_nr_a][model_nr_b]}', file=f)

    print('\nReminder: 1=offline LSTM; 2=offline+online LSTM; 3=offline LR; 4=online LR ', file=f)
print('Saved results to file.')
# %%

