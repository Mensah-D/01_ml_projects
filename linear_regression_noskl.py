#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 17:40:53 2026

@author: dennismensah
"""

# Import librarires
import matplotlib.pyplot as plt
import numpy as np

my_data = np.genfromtxt('/Users/dennismensah/Downloads/data.csv', delimiter = ',') # read the data
X = my_data [:, 0].reshape (-1,1) # -1 tells numpy to figure out the dimension by itself
ones = np.ones ([X.shape[0], 1]) # create a array containing only ones
X = np.concatenate([ones, X],1) # concatenate the ones to X matrix
y = my_data [:, 1].reshape(-1,1) # create the y matrix

#  Plot a chart
plt.scatter(my_data[:,0].reshape(-1,1), y)

# Values for Alpha + Theta
alpha = 0.0001
iters = 1000

theta = np.array([[1.0, 1.0]])

# Cost Function
def computeCost(X, y, theta):
    inner = np.power (((X @ theta.T) - y), 2) # @ means matrix multiplication of arrays.
    return np.sum(inner) / (2 * len(X))

computeCost(X, y, theta)

# Create Gradient Descent Function
def gradientDescent(X, y, theta, alpha, iters):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
        cost = computeCost(X, y, theta)
        # if i % 10 == 0: # just look at cost every ten loops for debugging
        # print(cost)
    return(theta, cost)
    
g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g, cost)

plt.scatter(my_data[:, 0].reshape(-1,1), y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = g[0][0] + g[0][1]* x_vals
plt.plot(x_vals, y_vals, '--')

