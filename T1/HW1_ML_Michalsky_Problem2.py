#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 23:47:41 2018

@author: filipmichalsky
"""

#####################
# CS 181, Spring 2018
# Homework 1, Problem 2
#
##################
import statsmodels.api as sm
from sklearn.metrics import r2_score
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()


# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T


# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(grid_X.T, w)

Y_hat = np.dot(X,w)
# TODO: plot and report sum of squared error for each basis

train_r2 = r2_score(Y, Y_hat)

print("Simple linear case training r-squared =", train_r2)

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.title("Simple linear regression, r^2: {:2f}".format(train_r2))
plt.show()

# Base a)
def make_Xa(years):
    years_multi = [years for i in range(1,6)]
    base = np.squeeze(years_multi)
    for i,j in enumerate(range(1,base.shape[0])):
        base[i]=np.power(base[i],j)
    base = np.squeeze(base)
    new_base = np.zeros([base.shape[1],5])
    for idx,j in enumerate(base):
        #print(base[idx].shape)
        new_base[:,idx]=base[idx]
    Xa = np.vstack((np.ones(years.shape),new_base.T)).T
    return Xa

Xa=make_Xa(years)
######

train_ols = sm.OLS(Y, Xa)
ols_fit = train_ols.fit()
#print(ols_fit.summary(),"\n")

print("Params:")
print(ols_fit.params, "\n")

print("conf_int:")
print(ols_fit.conf_int())

# TODO: plot and report sum of squared error for each basis
train_r2 = r2_score(Y, ols_fit.predict())



print("\nBase a) training r-squared =", train_r2)

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o')
plt.plot(years,ols_fit.predict())
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.title("Linear Reg Base a), r^2: {:1f}".format(train_r2))
plt.show()

# Base b)
def basis_b (x):
    return np.array([1] + [np.exp(-((x - mu_j)**2)/25)
                           for mu_j in range(1960, 2015, 5)])


Xb = np.array([basis_b(year) for year in years])
######

train_ols = sm.OLS(Y, Xb)
ols_fit = train_ols.fit()
#print(ols_fit.summary(),"\n")

print("Params:")
print(ols_fit.params, "\n")

print("conf_int:")
print(ols_fit.conf_int())

# TODO: plot and report sum of squared error for each basis
train_r2 = r2_score(Y, ols_fit.predict())

print("\nBase b) training r-squared =", train_r2)

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o')
plt.plot(years,ols_fit.predict())
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.title("Linear Reg Base b), r^2: {:1f}".format(train_r2))
plt.show()

def basis_c (x):
    return np.array([1] + [np.cos(x/j) for j in range(1, 6)])

Xc = np.array([basis_c(year) for year in years])
######

train_ols = sm.OLS(Y, Xc)
ols_fit = train_ols.fit()
#print(ols_fit.summary(),"\n")

print("Params:")
print(ols_fit.params, "\n")

print("conf_int:")
print(ols_fit.conf_int())

# TODO: plot and report sum of squared error for each basis
train_r2 = r2_score(Y, ols_fit.predict())

print("\nBase b) training r-squared =", train_r2)

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o')
plt.plot(years,ols_fit.predict())
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.title("Linear Reg Base c), r^2: {:1f}".format(train_r2))
plt.show()


def basis_d (x):
    return np.array([1] + [np.cos(x/j) for j in range(1, 26)])

Xd = np.array([basis_d(year) for year in years])
######

train_ols = sm.OLS(Y, Xd)
ols_fit = train_ols.fit()
#print(ols_fit.summary(),"\n")

print("Params:")
print(ols_fit.params, "\n")

print("conf_int:")
print(ols_fit.conf_int())

# TODO: plot and report sum of squared error for each basis
train_r2 = r2_score(Y, ols_fit.predict())

print("\nBase b) training r-squared =", train_r2)

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o')
plt.plot(years,ols_fit.predict())
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.title("Linear Reg Base d), r^2: {:1f}".format(train_r2))
plt.show()

# Deselect years larger than desired
x_sunspots = sunspot_counts[years<last_year]
y_republicans = republican_counts[years<last_year]

# Sort the indices
idx_sort = x_sunspots.argsort()
x_sunspots = x_sunspots[idx_sort]
y_republicans = y_republicans[idx_sort]

plt.scatter(x_sunspots,y_republicans)
    

# Nothing fancy for outputs.
Y = y_republicans.reshape(-1,1)
X = np.vstack((np.ones(x_sunspots.shape), x_sunspots)).T



print(X.shape,Y.shape)
# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!


#grid_Yhat  = np.dot(grid_X.T, w)

Y_hat = np.dot(X,w)
# TODO: plot and report sum of squared error for each basis

train_r2 = r2_score(Y, Y_hat)

print("Simple linear case training r-squared =", train_r2)

# Plot the data and the regression line.
plt.plot(x_sunspots, Y_hat,color='orange')
plt.xlabel("Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("Simple linear regression, r^2: {:2f}".format(train_r2))
plt.show()

# Deselect years larger than desired
x_sunspots = sunspot_counts[years<last_year]
y_republicans = republican_counts[years<last_year]

# Sort the indices
idx_sort = x_sunspots.argsort()
x_sunspots = x_sunspots[idx_sort]
y_republicans = y_republicans[idx_sort]

plt.scatter(x_sunspots,y_republicans)

Xa = make_Xa(x_sunspots)
#Xa = np.array([basis_a(sunspot) for sunspot in x_sunspots])
######
print("years",years)
print(y_republicans.shape,Xa.shape)
train_ols = sm.OLS(y_republicans, Xa)
ols_fit = train_ols.fit()
#print(ols_fit.summary(),"\n")

print("Params:")
print(ols_fit.params, "\n")

print("conf_int:")
print(ols_fit.conf_int())

# TODO: plot and report sum of squared error for each basis
train_r2 = r2_score(y_republicans, ols_fit.predict())

print("\nBase b) training r-squared =", train_r2)

# Plot the data and the regression line.
#plt.plot(years, republican_counts, 'o')

plt.plot(x_sunspots,ols_fit.predict(),color='orange')
plt.xlabel("Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("Linear Reg Base a), r^2: {:1f}".format(train_r2))
plt.show()

# Deselect years larger than desired
x_sunspots = sunspot_counts[years<last_year]
y_republicans = republican_counts[years<last_year]

# Sort the indices
idx_sort = x_sunspots.argsort()
x_sunspots = x_sunspots[idx_sort]
y_republicans = y_republicans[idx_sort]

plt.scatter(x_sunspots,y_republicans)

#Xa = make_Xa(x_sunspots)
Xc = np.array([basis_c(sunspot) for sunspot in x_sunspots])
######
train_ols = sm.OLS(y_republicans, Xc)
ols_fit = train_ols.fit()
#print(ols_fit.summary(),"\n")

print("Params:")
print(ols_fit.params, "\n")

print("conf_int:")
print(ols_fit.conf_int())

# TODO: plot and report sum of squared error for each basis
train_r2 = r2_score(y_republicans, ols_fit.predict())

print("\nBase b) training r-squared =", train_r2)

# Plot the data and the regression line.
#plt.plot(years, republican_counts, 'o')

plt.plot(x_sunspots,ols_fit.predict(),color='orange')
plt.xlabel("Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("Linear Reg Base c), r^2: {:1f}".format(train_r2))
plt.show()

# Deselect years larger than desired
x_sunspots = sunspot_counts[years<last_year]
y_republicans = republican_counts[years<last_year]

# Sort the indices
idx_sort = x_sunspots.argsort()
x_sunspots = x_sunspots[idx_sort]
y_republicans = y_republicans[idx_sort]

plt.scatter(x_sunspots,y_republicans)

#Xa = make_Xa(x_sunspots)
Xd = np.array([basis_d(sunspot) for sunspot in x_sunspots])
######
train_ols = sm.OLS(y_republicans, Xd)
ols_fit = train_ols.fit()
#print(ols_fit.summary(),"\n")

print("Params:")
print(ols_fit.params, "\n")

print("conf_int:")
print(ols_fit.conf_int())

# TODO: plot and report sum of squared error for each basis
train_r2 = r2_score(y_republicans, ols_fit.predict())

print("\nBase b) training r-squared =", train_r2)

# Plot the data and the regression line.
#plt.plot(years, republican_counts, 'o')

plt.plot(x_sunspots,ols_fit.predict(),color='orange')
plt.xlabel("Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("Linear Reg Base d), r^2: {:1f}".format(train_r2))
plt.show()





