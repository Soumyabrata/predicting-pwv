import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
from matplotlib.dates import DateFormatter
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
import pandas as pd
from read_matfile import *


# Input data
matlab_file = './data/PWV_2010from_WS_2_withGradient.mat'
(timestamp, pwv) = read_matfile (matlab_file)

# convert to pandas dataframe
datas = np.column_stack((timestamp, pwv))
df =  pd.DataFrame(data=datas, columns=['timestamps', 'pwv']).set_index(['timestamps'])


start_index = 73513 # or 83513 (to generate the second example)
train= df[start_index:start_index+10000]
test= df[start_index+10000:start_index+10000+50]



y_hat_avg = test.copy()
print ('computation started')
fit1 = ExponentialSmoothing(np.asarray(train['pwv']) ,seasonal_periods=288 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
print ('computation completed')


fig = plt.figure(1, figsize=(10,5))
plt.plot(train['pwv'][-150:], 'b:', label='Train')
plt.plot(test['pwv'], 'r--', label='Test')
plt.plot(y_hat_avg['Holt_Winter'], 'k-', label='Predicted')
plt.legend(loc='best', fontsize=18)
fig.autofmt_xdate()
formatter = DateFormatter('%d-%m-%y %H:%M')
plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.xlabel('Timestamps', fontsize=14)
plt.ylabel('Precipitible Water Vapor (in mm)', fontsize=14)

fig.tight_layout()
save_name = './results/73513-example.pdf' # or 83513 (to generate the second example)
fig.savefig(save_name)
plt.show()