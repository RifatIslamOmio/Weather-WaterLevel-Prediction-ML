import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

# global random seed
RAND_SEED = 42

def train_regression(model, param_grid, df, cls):
  """
  train regression model with GridSearchCV and 10-fold cross validation 
  to find the best hyper-parameters and training performance
  
  parameters- 
  @model: object of model class, 
  @param_grid: dictonary of hyper-parameters to be checked
  @X_train: features int the train dataset
  @y_train: output class in the train dataset

  returns- 
  @model: the best model found by GridSearchCV and 10-fold cross validation
  @selected_hyperparams: selected hyper-parameters for the best model
  @train_r2: mean r2 score during 10-fold cross validation
  @train_mae: mean mae score during 10-fold cross validation
  @train_rmse: mean rmse score during 10-fold cross validation
  """

  # 10-fold cross validation
  cv = KFold(n_splits=10, random_state=RAND_SEED, shuffle= True)

  X_train = df.drop(columns=cls)
  y_train = df[cls]

  # use gridsearch to check all values in param_grid
  model = GridSearchCV(model, param_grid, scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'], refit='r2', cv=cv)
  # fit model to data
  model.fit(X_train, y_train)

  selected_hyperparams = model.best_params_
  train_r2 = round(model.cv_results_['mean_test_r2'][model.best_index_], 4)
  train_mae = -1*round(model.cv_results_['mean_test_neg_mean_absolute_error'][model.best_index_], 4)
  train_rmse = -1*round(model.cv_results_['mean_test_neg_root_mean_squared_error'][model.best_index_], 4)

  return model, selected_hyperparams, train_r2, train_mae, train_rmse

def eval_regression(model, df, cls):
  """
  get performance of the selecte model on the test set

  parameters-
  @model: selected object of the Model class
  @X_test: features in the test dataset
  @y_test: output class in the test dataset

  returns-
  @test_r2: r2 score in the test set
  @test_mae: mae score in the test set
  @test_rmse: rmse score in the test set
  """

  X_test = df.drop(columns=cls)
  y_test = df[cls]

  y_test_pred = model.predict(X_test)
  test_r2 = round(r2_score(y_test, y_test_pred), 4)
  test_mae = round(mean_absolute_error(y_test, y_test_pred), 4)
  test_rmse = round(sqrt(mean_squared_error(y_test, y_test_pred)), 4)

  return test_r2, test_mae, test_rmse


def foo():
    print('hello from foo')
