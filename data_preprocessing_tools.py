# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# All the other columns which are the independent variables
X = dataset.iloc[:, :-1].values
# Last col which is the dependent variables
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
# Fills in missing data with mean/average, can specify median too 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# specify columns that are numerical
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# Encoding categorical data
# Encoding the Independent Variable - making dummy variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# CT takes in name, type of encoder and col number, remainder default is drop, passthrough retains all data in table 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
# Important to split before scaling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling (Normalization & Standardization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Fit(calculates mean & median & transform)
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
# Only required to transform
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)