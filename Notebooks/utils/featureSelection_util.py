import pandas as pd
import numpy as np
from sklearn.feature_selection import (SelectKBest, SequentialFeatureSelector,
                                       f_regression, mutual_info_regression, 
                                       mutual_info_classif, f_classif)
from sklearn.svm import LinearSVC, LinearSVR

def pearson_correlation_fs(_df, cls, threshold_corr=0.75):
    """
    function to check correlation of each pair of features a
    and discard the one from the pair with corr > 'threshold_corr' 
    among the pair, the one with lower corr with the 'cls' is dropped

    parameters-
    @_df: train dataset
    @cls: name of the class/output column
    @threshold_corr: correlation threshold for feature selection

    returns-
    @df: train dataset with the selected features
    @cols_to_drop: columns to drop from the train dataset
    """
    
    df = _df.copy()
    
    corr_matrix = df.corr()
    cols_to_drop = set() # keep only unique features
    
    # get the class column index
    for idx in range(len(corr_matrix.columns)):
        if corr_matrix.columns[idx]==cls :
            cls_col_idx = idx
            break
    
    # find the features to drop
    for col1_idx in range(len(corr_matrix.columns)):
        for col2_idx in range(col1_idx):
            if corr_matrix.columns[col1_idx] == cls or corr_matrix.columns[col2_idx] == cls:
                continue
                
            if abs(corr_matrix.iloc[col1_idx, col2_idx]) > threshold_corr:
                if abs(corr_matrix.iloc[col1_idx, cls_col_idx]) < abs(corr_matrix.iloc[col2_idx, cls_col_idx]): 
                    col_to_drop = corr_matrix.columns[col1_idx] 
                else:
                    col_to_drop = corr_matrix.columns[col2_idx]
                
                print(f'dropping {col_to_drop} from ({corr_matrix.columns[col1_idx]}, {corr_matrix.columns[col2_idx]})')
                
                cols_to_drop.add(col_to_drop)
    
    cols_to_drop = list(cols_to_drop)
    df.drop(columns=cols_to_drop)
    
    return df, cols_to_drop


def seleckKBest_fs(_df, cls, is_regression,
                   fixed_cols=['Station_Barisal', 'Station_Gazipur', 'Station_Rangpur', 'Station_Habiganj'], 
                   num_features=7, 
                   fs_method=mutual_info_regression):
    """
    function to select 'k' features with statistical evaluation

    parameters-
    @_df: train dataset
    @cls: name of the class/output column
    @fixed_cols: (do not pass anything)
    @num_features: number of features to be selected
    @fs_methods: (check sklearn 'SelectKBest' documentation or don't pass anything)

    returns-
    @df: train dataset with the selected features
    @cols_to_drop: columns to drop from the train dataset
    """

    df = _df.copy()

    fixed_cols.append(cls)
    X = df.drop(columns=fixed_cols)
    y = df[cls]
    
    if is_regression:
      fs_method = mutual_info_regression
    else:
      fs_method = mutual_info_classif

    # select top 'num_features' features based on mutual info regression
    # total features would be 'num_features' + 1(station column) 
    selector = SelectKBest(fs_method, k=num_features)
    selector.fit(X, y)
    selected_cols = list(X.columns[selector.get_support()])

    cols_to_drop = []
    for col in df.columns:
        if col in [cls, 'Station_Barisal', 'Station_Gazipur', 'Station_Rangpur', 'Station_Habiganj']:
            continue
        elif col not in selected_cols:
            cols_to_drop.append(col)
            
    df.drop(columns=cols_to_drop)
    
    return df, cols_to_drop


def selectSequential_fs(_df, cls, is_regression,
                        fixed_cols=['Station_Barisal', 'Station_Gazipur', 'Station_Rangpur', 'Station_Habiganj'], 
                        num_features=7, 
                        fs_method='forward'):
    """
    function to select 'k' features by evaluating performance on Linear SVM

    parameters-
    @_df: train dataset
    @cls: name of the class/output column
    @fixed_cols: (do not pass anything)
    @num_features: number of features to be selected
    @fs_methods: (check sklearn 'SequentialFeatureSelector' documentation or don't pass anything)

    returns-
    @df: train dataset with the selected features
    @cols_to_drop: columns to drop from the train dataset
    """

    df = _df.copy()

    fixed_cols.append(cls)
    X = df.drop(columns=fixed_cols)
    y = df[cls]
 
    if is_regression:
      estimator = LinearSVR(C=0.01, random_state=42)
      scoring='r2'
    else:
      estimator = LinearSVC(C=0.01, penalty="l1", dual=False, random_state=42)
      scoring = 'accuracy'
    
    # select top 'num_features' features based on mutual info regression
    # total features would be 'num_features' + 1(station column) 
    selector = SequentialFeatureSelector(estimator=estimator, n_features_to_select=num_features, cv=10, direction=fs_method, scoring=scoring)
    selector.fit(X, y)
    selected_cols = list(X.columns[selector.get_support()])

    cols_to_drop = []
    for col in df.columns:
        if col in [cls, 'Station_Barisal', 'Station_Gazipur', 'Station_Rangpur', 'Station_Habiganj']:
            continue
        elif col not in selected_cols:
            cols_to_drop.append(col)
            
    df.drop(columns=cols_to_drop)
    
    return df, cols_to_drop


def foo():
    print('hello from featureSelection_util foo')