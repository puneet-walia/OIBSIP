#Oasis Infobyte Project
##Sales Prediction
#"task 1"
###Arshpreet Singh

# Random Forest Regression

## Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset

dataset = pd.read_csv('Advertising.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(X)

## Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Training the Random Forest Regression model on the whole dataset

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

## Predicting the Test set results

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

## Evaluating the Model Performance

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
from sklearn.metrics import mean_squared_error
m = mean_squared_error(y_test,y_pred)
print(m)

print(X_test.shape)
print(y_test.shape)

plt.scatter(X_test[:,2], y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Test set')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
