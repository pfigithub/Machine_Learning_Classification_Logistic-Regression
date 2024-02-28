# import packages
import pandas as pd
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing

# readind data
churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()

# selection and pre-procesing
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

# seprate x and y
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

# pre-process
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)