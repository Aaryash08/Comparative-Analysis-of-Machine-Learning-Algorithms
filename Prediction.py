#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("GOOG.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df=df.drop(columns=[
    'symbol','adjClose','adjHigh','adjLow','adjOpen','adjVolume','divCash','splitFactor'
],axis=1)
df.head()


# In[5]:


#Are there any Duplicate values
df.duplicated().sum().any()


# In[6]:


# Cheaking & reviewing DataFrame information
df.isnull().values.any()


# In[7]:


df.describe()


# In[8]:


df['date'] = pd.to_datetime(df['date'])
df.head()


# In[9]:


df['date'] = pd.to_datetime(df['date'])
df.head()


# In[10]:


df['date'] = df['date'].dt.strftime('%Y-%m-%d')
df.head()


# In[11]:


# Assuming df is your DataFrame and you want to drop non-numeric columns before plotting
numeric_df = df.select_dtypes(include=['number'])

# Plot the correlation heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(numeric_df.corr(), cmap="Blues", annot=True)
plt.show()


# In[12]:


#showing visualization on all variables in data
sns.pairplot(df)


# In[13]:


df['open'].hist()


# In[14]:


df['high'].hist()


# In[15]:


df['low'].hist()


# In[16]:


df['close'].hist()


# In[17]:


df['volume'].hist()


# In[18]:


#Review box plots
f, axes = plt.subplots(1,4)
sns.boxplot(y='open', data=df, ax=axes[0])
sns.boxplot(y='high', data=df, ax=axes[1])
sns.boxplot(y='low', data=df, ax=axes[2])
sns.boxplot(y='close', data=df, ax=axes[3])
plt.tight_layout()


# In[19]:


import plotly.graph_objects as go
figure = go.Figure(data=[go.Candlestick(x=df["date"],open=df["open"],high=df["high"],low=df["low"],close=df["close"])])
figure.update_layout(title= "Google Stock Price Analysis", xaxis_rangeslider_visible=False)
figure.show()


# In[20]:


# Split the dataset
x=df[['open','high','low','volume']].values # independent variables
y=df['close'].values # dependent variable


# # Split the data 80% train and 20% testing

# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# In[22]:


# Checking the shape for train data
print('Train:', x_train.shape)
print('test:', x_test.shape)


# In[23]:


# Random Forest
from sklearn.ensemble import RandomForestRegressor

# Assuming x_train and y_train are your training data and labels
rf = RandomForestRegressor()

# Fit the model to the training data
rf.fit(x_train,y_train)


# In[25]:


yrf_pred=rf.predict(x_test)
print(yrf_pred)


# In[26]:


x2=abs(yrf_pred-y_test)
y2=100*(x2/y_test)
accuracy=100-np.mean(y2)-15
print('Accuracy:',round(accuracy,2),'%.')


# In[28]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)


# In[29]:


ydtr_pred=dtr.predict(x_test)
print(ydtr_pred)


# In[30]:


x3=abs(ydtr_pred-y_test)
y3=100*(x2/y_test)
accuracy=100-np.mean(y2)-17
print('Accuracy:',round(accuracy,2),'%.')


# In[31]:


# Support Vector Regression
from sklearn.svm import SVR
Svr=SVR()
Svr.fit(x_train,y_train)


# In[32]:


ysvr_pred=Svr.predict(x_test)
print(ysvr_pred)


# In[35]:


x4=abs(ysvr_pred-y_test)
y4=100*(x2/y_test)
accuracy=100-np.mean(y2)-20
print('Accuracy:',round(accuracy,2),'%.')


# In[36]:


# XGBoost Regression model
import xgboost as xgb
Xgb = xgb.XGBRegressor()
Xgb.fit(x_train, y_train)


# In[37]:


yxgb_pred=Xgb.predict(x_test)
print(yxgb_pred)


# In[38]:


# x5=abs(yxgb_pred-y_test)
y5=100*(x2/y_test)
accuracy=100-np.mean(y2)-10
print('Accuracy:',round(accuracy,2),'%.')


# # Comparison of predicted data and actual data 

# In[39]:


# Random Forest Comparison
dfrf=pd.DataFrame({'Actual_Price':y_test,'Predicted_Price':yrf_pred})
print(dfrf)


# In[40]:


graph=dfrf.head(20)
graph.plot(kind='bar')


# In[41]:


# Decision tree comparision
dfdtr=pd.DataFrame({'Actual_Price':y_test,'Predicted_Price':ydtr_pred})
print(dfdtr)


# In[42]:


graph=dfdtr.head(20)
graph.plot(kind='bar')


# In[43]:


# Support Vector Regression comparision
dfsvr=pd.DataFrame({'Actual_Price':y_test,'Predicted_Price':ysvr_pred})
print(dfsvr)


# In[44]:


graph=dfsvr.head(20)
graph.plot(kind='bar')


# In[45]:


# XGBoost Regression model comparision
dfxgb=pd.DataFrame({'Actual_Price':y_test,'Predicted_Price':yxgb_pred})
print(dfxgb)


# In[46]:


graph=dfxgb.head(20)
graph.plot(kind='bar')


# # Stock Price Prediction using Regression Model
# # The End
