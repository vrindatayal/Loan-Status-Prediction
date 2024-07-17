#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data Collection and Processing

# In[2]:


# loading the dataset to pandas DataFrame
loan_dataset = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')


# In[3]:


type(loan_dataset)


# In[4]:


# printing the first 5 rows of the dataframe
loan_dataset.head()


# In[5]:


# number of rows and columns
loan_dataset.shape


# In[6]:


# statistical measures
loan_dataset.describe()


# In[7]:


# number of missing values in each column
loan_dataset.isnull().sum()


# In[8]:


# dropping the missing values
loan_dataset = loan_dataset.dropna()


# In[9]:


# number of missing values in each column
loan_dataset.isnull().sum()


# In[10]:


loan_dataset.head()


# In[11]:


# label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[12]:


# printing the first 5 rows of the dataframe
loan_dataset.head()


# In[13]:


# Dependent column values
loan_dataset['Dependents'].value_counts()


# In[14]:


# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)


# In[15]:


# Dependent column values
loan_dataset['Dependents'].value_counts()


# In[16]:


# Dependent column values
loan_dataset['Gender'].value_counts()


# In[17]:


# Dependent column values
loan_dataset['Married'].value_counts()


# In[18]:


loan_dataset['Education'].value_counts()


# In[19]:


loan_dataset['Self_Employed'].value_counts()


# In[20]:


loan_dataset['Credit_History'].value_counts()


# In[21]:


loan_dataset['Property_Area'].value_counts()


# In[22]:


# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1}},inplace=True)
loan_dataset.replace({'Gender':{'Male':1,'Female':0}},inplace=True)
loan_dataset.replace({'Self_Employed':{'No':0,'Yes':1}},inplace=True)
loan_dataset.replace({'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True)
loan_dataset.replace({'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)


# In[23]:


loan_dataset.head()


# Data Visualisation

# In[24]:


# education & Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)


# In[25]:


# marital status & Loan Status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)


# In[26]:


# separating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']


# In[27]:


print(X)
print(Y)


# Test train split

# In[28]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)


# In[29]:


print(X.shape, X_train.shape, X_test.shape)


# Training the model:
# 
# Support Vector Machine Model

# In[30]:


classifier = svm.SVC(kernel='linear')


# In[31]:


#training the support Vector Macine model
classifier.fit(X_train,Y_train)


# Model Evaluation

# In[32]:


# accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,Y_train)


# In[33]:


print('Accuracy on training data : ', training_data_accuray)


# In[34]:


# accuracy score on training data
X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)


# In[35]:


print('Accuracy on test data : ', test_data_accuray)

