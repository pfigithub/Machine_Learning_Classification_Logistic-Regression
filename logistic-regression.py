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

# train-test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# modeling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
# predicing
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)