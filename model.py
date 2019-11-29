# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 06:20:07 2019

@author: Anjana
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 05:38:12 2019

@author: Anjana
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")

data=pd.read_csv('housing.csv')
prices= data['MEDV']
features=data.drop('MEDV', axis=1)

print("Boston housing dataset has {} data points with {} variables each".format(*data.shape))

from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):

  score=r2_score(y_true,y_predict)

  return score

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=42)

print("Training and testing split was successful.")
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def fit_model(X,y):

  cv_sets=ShuffleSplit(n_splits=10, test_size=0.20, random_state=0 )
  regressor=DecisionTreeRegressor()

  params={'max_depth':list(range(1,11))}
  scoring_fnc=make_scorer(performance_metric)

  grid=GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

  grid=grid.fit(X,y)
  depths= [d['max_depth'] for d in grid.cv_results_["params"]]
  scores=grid.cv_results_["mean_test_score"]
  df=pd.DataFrame({"max_depth":depths, "mean_test_score":scores}, columns=["max_depth", "mean_test_score"])

  return grid.best_estimator_, df

from IPython.core.display import HTML

reg, grid_table= fit_model(X_train, y_train)

reg.predict([[1,2,3],[12,14,16],[30,70,9]])

import pickle
filename= 'bostondatasetmodel.sav'
pickle.dump(reg, open(filename, 'wb'))

