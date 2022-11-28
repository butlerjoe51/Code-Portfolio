#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import pandas as pd
import datetime
import pandas_datareader.data as web

#Remove display limits on Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


btc = web.get_data_yahoo(['BTC-USD'], start=datetime.datetime(2018, 1, 1), 
end=datetime.datetime(2020, 12, 2))['Close']

print(btc.head())


# In[4]:


#Convert Data to CSV file
btc.to_csv("btc.csv")
btc = pd.read_csv("btc.csv")
print(btc.head())


# In[5]:


#Make the date column a data frame index to allow models to be used
btc.index = pd.to_datetime(btc['Date'], format='%Y-%m-%d')
del btc['Date']


# In[6]:


print(btc.head())


# In[8]:


#Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.title('Time Series of BTC between Jan 2018 and Dec 2020')
plt.xticks(rotation=45)

plt.plot(btc.index, btc['BTC-USD'],)


# In[9]:


#Split Data for Testing and Training
train = btc[btc.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')] #Data before Nov 2020 is train
test = btc[btc.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')] #Data after Nov 2020 is test

plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for BTC Data")
plt.show()


# In[11]:


# ARMA model - uses past values to predict future values
from statsmodels.tsa.statespace.sarimax import SARIMAX
y = train['BTC-USD']
ARMAmodel = SARIMAX(y, order = (1,0,1)) #ARMA model with SARIMAX class with 95% confidence interval
ARMAmodel = ARMAmodel.fit() #Fit model to prepare for predictions


# In[12]:


#Generate Predictions
y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 


# In[16]:


#Plot Results
plt.plot(train, color = "black", label = 'Training')
plt.plot(test, color = "red", label = 'Testing')
plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for BTC Data")
plt.legend()


# In[17]:


#Evaulate performance using root mean-squared error
import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["BTC-USD"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)


# In[22]:


#RMS Error is very high - over 3700
#Can we improve performance?

#Try ARIMA!
#Import and prepare model
from statsmodels.tsa.arima.model import ARIMA
ARIMAmodel = ARIMA(y, order = (5, 4, 2))
ARIMAmodel = ARIMAmodel.fit()

#Plot Model
y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 

plt.plot(train, color = "black", label = 'Training')
plt.plot(test, color = "red", label = 'Testing')
plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
plt.legend()


#Mean Squared Error

arma_rmse = np.sqrt(mean_squared_error(test["BTC-USD"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)


# In[24]:


# RMSE is 886 - Much better. Can we do even better?
# Try SARIMA because it incorporates seasonality 

SARIMAXmodel = SARIMAX(y, order = (5, 4, 2), seasonal_order=(2,2,2,12))
SARIMAXmodel = SARIMAXmodel.fit()


# In[25]:


y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(train, color = "black", label = 'Training')
plt.plot(test, color = "red", label = 'Testing')
plt.plot(y_pred_out, color='Blue', label = 'SARIMA Predictions')
plt.legend()


# In[ ]:




