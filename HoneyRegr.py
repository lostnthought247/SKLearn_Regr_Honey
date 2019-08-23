#!/usr/bin/env python3
"""Runs a linear regression model in SKLearn to predict honey production by year based on past year's trends

Based on CodeAcademy Supervised Machine Learning exercise"""

# Import Required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Create initial dataframe from AWS S3 Bucket url
df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

# transform dataframe to list production value by year
prod_per_year = df.groupby("year").totalprod.mean().reset_index()

# Sets X,y variables and reshapes array to be SKLearn compatable
# .reshape(-1,1) transforms data from 1 row of data to 1 column of data
X = prod_per_year.year.values.reshape(-1,1)
y = prod_per_year.totalprod

# Create the linear regression model
regr = linear_model.LinearRegression()

# Fits the model with the input dataset
regr.fit(X, y)

# # print the slow and intercept of the fit linier model
# print(regr.coef_[0])
# print(regr.intercept_)

# Create a list of y predictions for each X datapoint
y_predict = regr.predict(X)

# Creates Y predictions for specific input future dates x
print("-------------------------------------------")
input_x = int(input("Please input a future year: "))
y_input_predict = regr.predict(input_x)
if y_input_predict < 0:
    y_input_predict = 0
print("...")
# Formats and displays terminal notifications
print("The anticipated honey production in %s will be: %s" % (input_x, y_input_predict))
if y_input_predict > 0:
    percent_dif_raw = 100* (regr.predict(2010) - regr.predict(input_x))/regr.predict(2010)
    percent_dif = round(float(''.join(map(str, percent_dif_raw))),2)
else:
    percent_dif_raw = 100* (regr.predict(2010) - 0)/regr.predict(2010)
    percent_dif = round(float(''.join(map(str, percent_dif_raw))),2)
print(" That is %s percent below 2010's production" % percent_dif)

# Creates an array of years for future dates spanning 2013-2050
X_future = np.array(range(2013, 2051))

# Reshape the array to work with SKLearn.
X_future = X_future.reshape(-1, 1)

# Creates Y predictions for input future dates x
future_predict = regr.predict(X_future)


# Display all original datapoints (X) and targets (y) as scatter graph
plt.scatter(X,y)
# Displays y prediction for input value X
plt.plot(input_x, y_input_predict, 'rX')
# Display linier regression modoel predictions for X as line plot
plt.plot(X, y_predict)
# predicts future x/y values based on linear trends in training data
plt.plot(X_future, future_predict)
# Display both scatter and plot charts
plt.show()
