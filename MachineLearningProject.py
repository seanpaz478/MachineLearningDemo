# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:07:01 2020

@author: Sean Paz
Machine learning project comparing the accuracy of multiple models to 
classify tumors as malignant or benign
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

# Importing dataset into pandas dataframe
masses = pd.read_csv('mammographic_masses.data.txt', na_values=['?'], names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
print(masses.describe())

# Cleaning out missing data (after checking that it's randomly distributed)
masses.dropna(inplace = True)

# Putting data into numpy arrays
features = masses[['age', 'shape', 'margin', 'density']].values
classes = masses['severity'].values
feature_names = ['age', 'shape', 'margin', 'density']

# Normalizing data for later models
scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit_transform(features)

# Utilizing decision trees

# Creating train and test data for a more accurate model
(x_inputs, y_inputs, x_classes, y_classes) = train_test_split(scaled_features, classes, train_size = 0.75, random_state = 12)

DT = DecisionTreeClassifier(random_state = 12)

DT.fit(x_inputs, x_classes)

DT_score = cross_val_score(DT, scaled_features, classes, cv = 10).mean()

print("Decision tree score: {score}".format(score = DT_score))

DTF = RandomForestClassifier(n_estimators = 10, random_state = 12)
DTF_score = cross_val_score(DTF, scaled_features, classes, cv = 10).mean()

print("Decision tree forest score: {score}".format(score = DTF_score))

# Utilizing a K-Nearest-Neighbors model

print('KNN scores for different values of k:\n')
highest_mean = 0

# Using a loop to determine what the best value for k would produce the best score
for n in range(1, 101):
    KNN = neighbors.KNeighborsClassifier(n_neighbors = n)
    KNN_scores = cross_val_score(KNN, scaled_features, classes, cv = 10)
    print("{k} {score}".format(k = n, score = KNN_scores.mean()))
    if KNN_scores.mean() > highest_mean:
        highest_mean = KNN_scores.mean()
        best_k = n
print("Best value for k and KNN score: {n} {score}".format(n = best_k, score = highest_mean))
    
# Using an SVM model with a linear kernel 
C = 1.0
linear_svc = svm.SVC(kernel = 'linear', C = C)
linear_svc_scores = cross_val_score(linear_svc, scaled_features, classes, cv = 10)
print("Linear SVC score: {score}".format(score = linear_svc_scores.mean()))

# Using an SVM model with a 3rd degree polynomial kernel
poly_svc = svm.SVC(kernel = 'poly', degree = 3, gamma = 'scale')
poly_svc_scores = cross_val_score(poly_svc, scaled_features, classes, cv = 10)
print("Polynomial SVC score: {score}".format(score = poly_svc_scores.mean()))

# Using a Multinomial Naive Bayes model
minmax_scaler = preprocessing.MinMaxScaler()
minmax_features = minmax_scaler.fit_transform(features)
MNB = MultinomialNB()
cv_scores = cross_val_score(MNB, minmax_features, classes, cv = 10)
print("Multinomial Naive Bayes score: {score}".format(score = cv_scores.mean()))

