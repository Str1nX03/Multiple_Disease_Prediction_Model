# %%
# Importing  necessary libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
# Data Collection

heart_data = pd.read_csv('heart.csv')

# %%
# Read the dataset

heart_data.head()

# %%
# Dimension of the dataset

heart_data.shape

# %%
# Getting some information about the data

heart_data.info()

# %%
# Checking for missing values

heart_data.isnull().sum()

# %%
# Statistical measures of the data

heart_data.describe()

# %%
# Checking the distribution of the target variable

heart_data['target'].value_counts()

# %%
# Splitting features and target

X = heart_data.drop(columns = 'target' , axis = 1)
Y = heart_data['target']

# %%
# Checking feature

print(X)

# %%
# Checking target

print(Y)

# %%
# Splitting the data into training and testing data

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2 , stratify = Y , random_state = 2)

# %%
# Initiating Logistic Regression algorithm

model = LogisticRegression()

# %%
# Fitting data into the model

model.fit(X_train , Y_train)

# %%
# Model evaluation on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)

# %%
# Lets check the accuracy score on training data

print("Accuracy on Training Data: " , training_data_accuracy)

# %%
# Model evaluation on testing data

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction , Y_test)

# %%
# Lets check the accuracy score testing data

print("Accuracy on Testing Data: " , testing_data_accuracy)

# %%
# Building a predictive system

input_data = (62,0,0,160,164,0,0,145,0,6.2,0,3,3)
input_data_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)

if prediction[0] == 1:

    print('The person have the heart disease.')

else:

    print("The person doesn't have the heart disease.")

# %%
# Importing pickle library

import pickle

# %%
# Saving the trained model

filename = 'heart_disease_model.sav'
pickle.dump(model , open(filename , 'wb'))

# %%
