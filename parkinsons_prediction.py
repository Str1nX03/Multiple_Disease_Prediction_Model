# %%
# Importing required Library

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# %%
# Loading the data from csv file to a Pandas DataFrame

parkinsons_data = pd.read_csv('parkinsons.csv')

# %%
# Printing the first 5 rows of the dataframe

parkinsons_data.head()

# %%
# Getting more information about the dataset

parkinsons_data.info()

# %%
# Checking for missing values in each column

parkinsons_data.isnull().sum()

# %%
# Getting some statistical measures about the data

parkinsons_data.describe()

# %%
# Distribution of target Variable

parkinsons_data['status'].value_counts()

# %%
# Separating feature and target class

X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']

# %%

print(X)

# %%

print(Y)

# %%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# %%

model = svm.SVC(kernel='linear')

# %%
# Training the SVM model with training data

model.fit(X_train, Y_train)

# %%
# Accuracy score on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# %%

print('Accuracy score of training data : ', training_data_accuracy)

# %%
# Accuracy score on training data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# %%
# Building a Predictive System

input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")

# %%
# Saving the trained model

import pickle
     
filename = 'parkinsons_model.sav'
pickle.dump(model, open(filename, 'wb'))
