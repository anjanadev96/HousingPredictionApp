# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 06:21:59 2019

@author: Anjana
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pickle

from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('web.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    rm=request.form.get("RM")
    lstat=request.form.get("LSTAT")
    ptratio=request.form.get("PTRATIO")
    
    filename= 'bostondatasetmodel.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    x=str(loaded_model.predict([[int(rm), int(lstat),int(ptratio)]]))
    return "Predicted value of the house is"+x

if __name__=="__main__": 
    app.run(port=5000,debug=False)

    