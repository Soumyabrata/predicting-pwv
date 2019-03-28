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


lead_time_array = np.arange(5, 30, 5)
previous_time_array = np.arange(12000, 2000, -2000)
no_of_experiments = 10

rmse_matrix = np.zeros([len(previous_time_array), len(lead_time_array)])
print (rmse_matrix)



for i, item1 in enumerate(lead_time_array):
    for j, item2 in enumerate(previous_time_array):


        lead_time = item1
        previous_time = item2

        lead_observations = int(lead_time / 5)
        previous_observations = int(previous_time / 5)


        rmse_array = []
        for _ in range(no_of_experiments):

            last_possible_index = end_index_of_df - (previous_observations+lead_observations)
            start_index = random.randint(0, last_possible_index)
            print ('From start index of ', str(start_index))

            print (['computing for lead time = ', str(lead_time), ' mins with history of ', str(previous_time), ' mins'])
            train= df[start_index:start_index+previous_observations]
            test= df[start_index+previous_observations:start_index+previous_observations+lead_observations]
            y_hat_avg = test.copy()
            print ('computation started')
            fit1 = ExponentialSmoothing(np.asarray(train['pwv']) ,seasonal_periods=288 ,trend='add', seasonal='add',).fit()
            y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
            print ('computation completed')


            # computing the error
            a = y_hat_avg['pwv']
            b = y_hat_avg['Holt_Winter']
            rmse_value = np.sqrt(np.mean((b-a)**2))

            rmse_array.append(rmse_value)

        rmse_array = np.array(rmse_array)
        rmse_matrix[j,i] = np.mean(rmse_array)

        print (rmse_matrix)


np.save('./results/rmse_matrix_for_grid.npy', rmse_matrix)

# Plotting and visualizing the results
(no_of_y_components, no_of_x_components) =  rmse_matrix.shape
xlabels = []
for i in range(no_of_x_components):
    xlabels.append(5*(i+1))

ylabels = []
current_index = previous_time_array[0]
diff = previous_time_array[0] - previous_time_array[1]
for i in range(no_of_y_components):
    ylabels.append(current_index)
    current_index = current_index - diff

fig, ax = plt.subplots()
cax = ax.imshow(rmse_matrix, cmap=plt.cm.coolwarm)
plt.xticks([]),plt.yticks([])
plt.xticks(np.arange(0,no_of_x_components,1), xlabels)
plt.yticks(np.arange(0,no_of_y_components,1), ylabels)
plt.xlabel('Lead Times (in mins)', fontsize=12)
plt.ylabel('Historical Data (in mins)', fontsize=12)
cbar = fig.colorbar(cax, ticks=[rmse_matrix.min(), rmse_matrix.max()], orientation='vertical')
cbar.ax.set_yticklabels(['Low', 'High'])  # horizontal colorbar
fig.tight_layout()
fig.savefig('./results/rmse.pdf')
plt.show()