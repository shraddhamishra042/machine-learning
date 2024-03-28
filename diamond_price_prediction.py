#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# In[2]:


diamond = pd.read_csv(r"C:\Users\msaur\Downloads\diamonds.csv")
diamond.head()


# In[3]:


diamond.info()  


# In[4]:


diamond.describe()  


# In[5]:


diamond.shape  #with the help of shape we can extract number of rows and columns


# In[6]:


diamond.isnull().sum()   #missing values are not present


# In[7]:


#seperating x and y
x=diamond.iloc[:,0:1] #2d
y=diamond.iloc[:,-1]  #1d


# In[8]:


x


# In[9]:


y


# In[10]:


#split data into training and testing


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=1)


# In[13]:


xtrain.shape


# In[14]:


#building a machine learning model


# In[15]:


#step1: import model
from sklearn.linear_model import LinearRegression

#step2: create an instance of a model
lr=LinearRegression()

#step3: train a model
lr.fit(xtrain,ytrain)

#step4: predict values
ypred=lr.predict(xtest)


# In[16]:


xtest


# In[17]:


ypred  #redicted value


# In[18]:


ytest # actual value 


# In[19]:


#differnce between actual and predicted value


# In[20]:


#slope (mean(value of m and c))
lr.coef_


# In[21]:


#intercept value
lr.intercept_


# In[22]:


#evaluvating the model
from sklearn.metrics import r2_score   

r2_score(ytest,ypred)


# In[23]:


#predicting new observation and user define observation


# In[24]:


lr.predict([[4.4]])  #1d array is not valid


# In[ ]:




