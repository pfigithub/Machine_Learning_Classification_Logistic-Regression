# import packages
import pandas as pd
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing

# readind data
churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()

