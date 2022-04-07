import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import label_binarize

# global random seed
RAND_SEED = 42

def train_regression(model, param_grid, df, cls):
  """
  train regression model with GridSearchCV and 10-fold cross validation 
  to find the best hyper-parameters and training performance
  
  parameters- 
  @model: object of model class, 
  @param_grid: dictonary of hyper-parameters to be checked
  @df: train dataset
  @cls: output label

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
  @df: test dataset

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


def showEvalutationGraph_regression(ModelClass, df, cls, 
                                    x_axis_param_name, x_axis_param_vals, selected_model_params):
  """
  generate graph with r2 scores of the model on the train set
  for different values of a hyper-parameter

  parameters-
  @model: the Model class itself (not an object)
  @df: features in the train dataset
  @cls: output class in the train dataset
  @x_axis_param_name: the hyper-parameter name to be shown on the x_axis
  @x_axis_param_vals: list of hyper-parameter values
  @selected_model_params: dictonary of parameters for the model selected by GridSearchCV
                          also pass in parameters like 'random_state' and 'n_jobs' if required 
  """

  X = df.drop(columns=cls)
  y = df[cls]
  cv = KFold(n_splits=10, random_state=RAND_SEED, shuffle= True)

  r2s = []

  selected_x_axis_param_val = selected_model_params[x_axis_param_name]

  # sort the x axis parameters so the graph looks appropriate
  x_axis_param_vals.sort()

  for x_axis_param_val in x_axis_param_vals:
    model_params = selected_model_params
    model_params[x_axis_param_name] = x_axis_param_val

    model = ModelClass(**model_params)

    r2_segments = cross_val_score(model, X, y, scoring='r2',cv=cv, n_jobs=1)
    r2s.append(np.mean(r2_segments))

  plt.figure(figsize =(15,9))
  plt.plot(x_axis_param_vals, r2s, 'ro-')
  plt.axvline(x=selected_x_axis_param_val, color='k', linestyle='--')
  plt.legend(['R2-score', f'selected value ({x_axis_param_name}={selected_x_axis_param_val})'], fontsize=16)
  plt.xlabel(x_axis_param_name, fontsize=18)
  plt.ylabel('R2-score', fontsize=18)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.show()


def _under_sampling_strategy(y_train):
  freq = dict(Counter(y_train))
  # under sample ~20% of the majority class
  freq[0] = int(freq[0]*0.8)
  return freq

def train_classification(model, param_grid, df, cls, sampling_technique=None):
  """
  train classification model with Sampling technique and 
  GridSearchCV + 10-fold cross validation 
  
  parameters- 
  @model: object of model class, 
  @param_grid: dictonary of hyper-parameters to be checked
  @df: train dataset
  @cls: output label
  @sampling_technique: must be 'smote', 'rando', or 'hybrid'

  returns- 
  @model: the best model found by GridSearchCV and 10-fold cross validation
  @selected_hyperparams: selected hyper-parameters for the best model
  @train_accuracy: mean accuracy score during 10-fold cross validation
  @train_f1: mean macro-f1 score during 10-fold cross validation
  """

  X_train = df.drop(columns=cls)
  y_train = df[cls]

  if sampling_technique in ['smote', 'rando', 'hybrid']:
    print(f'class distribution before sampling: {dict(Counter(y_train))}')

  if sampling_technique=='smote':
    # apply SMOTE
    smote = SMOTE(random_state=RAND_SEED)
    X_train, y_train = smote.fit_resample(X_train, y_train)

  elif sampling_technique=='rando':
    # apply SMOTE
    rando = RandomOverSampler(random_state=RAND_SEED)
    X_train, y_train = rando.fit_resample(X_train, y_train)
    
  elif sampling_technique=='hybrid':
    # apply RandomUnderSampler + SMOTE
    randu = RandomUnderSampler(random_state=RAND_SEED, sampling_strategy=_under_sampling_strategy)
    X_train, y_train = randu.fit_resample(X_train, y_train)
    smote = SMOTE(random_state=RAND_SEED)
    X_train, y_train = smote.fit_resample(X_train, y_train)

  elif sampling_technique is not None:
    print(f"invalid sampling_technique='{sampling_technique}' passed. MUST be one of the following: ['smote', 'rando', 'hybrid']")

  if sampling_technique in ['smote', 'rando', 'hybrid']:
    print(f'class distribution after sampling: {dict(Counter(y_train))}')

  # 10-fold cross validation
  cv = StratifiedKFold(n_splits=10, random_state=RAND_SEED, shuffle= True)

  # use gridsearch to check all values in param_grid
  model = GridSearchCV(model, param_grid, scoring=['accuracy', 'f1_macro'], refit='accuracy', cv=cv)
  # fit model to data
  model.fit(X_train, y_train)

  selected_hyperparams = model.best_params_
  train_accuracy = round(model.cv_results_['mean_test_accuracy'][model.best_index_], 4)
  train_f1 = round(model.cv_results_['mean_test_f1_macro'][model.best_index_], 4)
  # train_auc = round(model.cv_results_['mean_test_roc_auc'][model.best_index_], 4)

  return model, selected_hyperparams, train_accuracy, train_f1


def eval_classification(model, df, cls):
  """
  get performance of the selecte model on the test set

  parameters-
  @model: selected object of the Model class
  @df: test dataset

  returns-
  @test_accuracy: accuracy score in the test set
  @test_f1: macro-f1 score in the test set
  """

  X_test = df.drop(columns=cls)
  y_test = df[cls]

  y_test_pred = model.predict(X_test)
  test_accuracy = round(accuracy_score(y_test, y_test_pred), 4)
  test_f1 = round(f1_score(y_test, y_test_pred, average='macro'), 4)
  
  y_test_proba = label_binarize(y_test, classes=[0, 1, 2])
  y_test_pred_proba = label_binarize(y_test_pred, classes=[0, 1, 2])
  test_auc = round(roc_auc_score(y_test_proba, y_test_pred_proba, multi_class='ovr'), 4)

  return test_accuracy, test_f1, test_auc


def showEvalutationGraph_classification(ModelClass, df, cls, 
                                        x_axis_param_name, x_axis_param_vals, selected_model_params):
  """
  generate graph with accuracy, macro-f1 scores of the model on the train set
  for different values of a hyper-parameter

  parameters-
  @model: the Model class itself (not an object)
  @df: features in the train dataset
  @cls: output class in the train dataset
  @x_axis_param_name: the hyper-parameter name to be shown on the x_axis
  @x_axis_param_vals: list of hyper-parameter values
  @selected_model_params: dictonary of parameters for the model selected by GridSearchCV
                          also pass in parameters like 'random_state' and 'n_jobs' if required 
  """

  X = df.drop(columns=cls)
  y = df[cls]
  
  cv = StratifiedKFold(n_splits=10, random_state=RAND_SEED, shuffle= True)

  accuracies = []
  f1s = []

  selected_x_axis_param_val = selected_model_params[x_axis_param_name]
    
  # sort the x axis parameters so the graph looks appropriate
  x_axis_param_vals.sort()

  for x_axis_param_val in x_axis_param_vals:
    model_params = selected_model_params
    model_params[x_axis_param_name] = x_axis_param_val

    model = ModelClass(**model_params)

    accuracy_segments = cross_val_score(model, X, y, scoring='accuracy',cv=cv, n_jobs=1)
    f1_segments = cross_val_score(model, X, y, scoring='f1_macro',cv=cv, n_jobs=1)
    # auc_segments = cross_val_score(model, X, y, scoring='roc_auc',cv=cv, n_jobs=1)
    accuracies.append(np.mean(accuracy_segments))
    f1s.append(np.mean(f1_segments))
    # aucs.append(np.mean(auc_segments))

  plt.figure(figsize =(15,9))
  plt.plot(x_axis_param_vals, accuracies, 'ro-',  x_axis_param_vals, f1s ,'bv-') #, x_axis_param_vals, aucs,'yo-')
  plt.axvline(x=selected_x_axis_param_val, color='k', linestyle='--')
  plt.legend(['Accuracy','Macro F1', f'selected value ({x_axis_param_name}={selected_x_axis_param_val})'], fontsize=16)
  plt.xlabel(x_axis_param_name, fontsize=18)
  plt.ylabel('Accuracy, Macro F1', fontsize=18)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.show()
