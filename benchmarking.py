import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
from matplotlib.dates import DateFormatter
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
import pandas as pd
import random
from read_matfile import *

matlab_file = './data/PWV_2010from_WS_2_withGradient.mat'
(timestamp, pwv) = read_matfile (matlab_file)
print ('Imported the MATLAB file')


# convert to pandas dataframe
datas = np.column_stack((timestamp, pwv))
df =  pd.DataFrame(data=datas, columns=['timestamps', 'pwv']).set_index(['timestamps'])
end_index_of_df = len(df)


lead_time_array = np.arange(5,20,5)
no_of_experiments = 10

previous_time = 10000
previous_observations = int(previous_time / 5)

text_file = open("./results/comparison.txt", "w")
text_file.write("time, our, naive, average \n")


for item1 in lead_time_array:
    lead_time = item1
    lead_observations = int(lead_time / 5)

    rmse_array = []
    persist_array = []
    average_array = []

    for _ in range(no_of_experiments):
        last_possible_index = end_index_of_df - (previous_observations + lead_observations)
        start_index = random.randint(0, last_possible_index)
        print('From start index of ', str(start_index))

        print(['computing for lead time = ', str(lead_time), ' mins with history of ', str(previous_time), ' mins'])
        train = df[start_index:start_index + previous_observations]
        test = df[start_index + previous_observations:start_index + previous_observations + lead_observations]
        y_hat_avg = test.copy()
        print('computation started')
        fit1 = ExponentialSmoothing(np.asarray(train['pwv']), seasonal_periods=288, trend='add', seasonal='add', ).fit()
        y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))

        # persistence
        last_value = train['pwv'][-1]
        y_hat_avg['naive'] = last_value * np.ones(len(test))

        # average
        mean_training_value = np.mean(train['pwv'])
        y_hat_avg['aver'] = mean_training_value*np.ones(len(test))

        print('computation completed')

        # computing the error for exponential smoothing
        a = y_hat_avg['pwv']
        b = y_hat_avg['Holt_Winter']
        rmse_value = np.sqrt(np.mean((b - a) ** 2))
        rmse_array.append(rmse_value)

        # computing the error for persistence model
        a = y_hat_avg['pwv']
        b = y_hat_avg['naive']
        rmse_value = np.sqrt(np.mean((b - a) ** 2))
        persist_array.append(rmse_value)

        # computing the error for average model
        a = y_hat_avg['pwv']
        b = y_hat_avg['aver']
        rmse_value = np.sqrt(np.mean((b - a) ** 2))
        average_array.append(rmse_value)



    rmse_array = np.array(rmse_array)
    persist_array = np.array(persist_array)
    average_array = np.array(average_array)

    text_file.write("%s, %s, %s, %s \n" % (lead_time, np.mean(rmse_array), np.mean(persist_array), np.mean(average_array)))


text_file.close()