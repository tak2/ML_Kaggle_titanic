# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# https://github.com/mrdbourke/your-first-kaggle-submission/blob/master/kaggle-titanic-dataset-example-submission-workflow.ipynb
import os

test = os.getcwd()
print(test)
train1 = os.getcwd() +'\\titanic\\train.csv'
test1 =  os.getcwd() +'\\titanic\\test.csv'
print (test1)

import pandas as pd

train = pd.read_csv(train1)
test = pd.read_csv(test1)

train.head(15)
# %%

for col in train.columns:
    print(col)
    
# %%
train.Age.plot.hist()

test.head()    
train.describe()

pd.set_option('display.max_columns', 100) 


# %% ## %%

train.Age.plot.hist()
train.describe()
# %% 

import missingno
print(dir(missingno))

# Plot graphic of missing values
missingno.matrix(train, figsize = (15,5))

# %% 
train.isnull().sum()

train.dtypes

import matplotlib.pyplot as plt

import seaborn as sns
plt.style.use('seaborn-whitegrid')

# How many people survived?
fig = plt.figure(figsize=(20,1))
sns.countplot(y='Survived', data=train);
print(train.Survived.value_counts())


sns.distplot(train.Pclass)


# %% 

df_bin = pd.DataFrame() # for discretised continuous variables
df_con = pd.DataFrame() # for continuous variables
# %% 


train.dtypes

# Let's add this to our subset dataframes
df_bin['Survived'] = train['Survived']
df_con['Survived'] = train['Survived']


df_bin.head()

df_bin['Pclass'] = train['Pclass']
df_con['Pclass'] = train['Pclass']

# %% 
import numpy as np

# add Sex to the subset dataframes
df_bin['Sex'] = train['Sex']
df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0) # change sex to 0 for male and 1 for female

df_con['Sex'] = train['Sex']

# %% 

# How does the Sex variable look compared to Survival?
# We can see this because they're both binarys.
fig = plt.figure(figsize=(10, 10))
sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'});
sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'label': 'Did not survive'});


# %% 

train.Age.isnull().sum()



# %% 
train.Age.mean()
train.Age.median()


# %% 

df_bin['Age'] = train.Age.replace(np.nan, train.Age.median())



# %% 
df_bin['Age'].heads()

# %% 
# %% 
# %% 
# %% 
# %% 
# %% 
# %% 
# %% 
# %% 
# %% 
# %% 
# %% 
# %% 