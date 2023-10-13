import warnings
warnings.filterwarnings('ignore')

import itertools
import textwrap

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
#os.environ["PYTHONWARNINGS"] = "default"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["RAY_LOG_TO_STDERR"] = '3'

import logging
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("worker").setLevel(logging.ERROR)

import numpy as np
np.seterr(all="ignore")

import sklearn
from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder, StandardScaler, KBinsDiscretizer, RobustScaler, QuantileTransformer
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,  GradientBoostingClassifier,  GradientBoostingRegressor

from catboost import CatBoostClassifier, CatBoostRegressor, Pool

import category_encoders as ce
import torch
import datasets
import pickle

from scipy.io.arff import loadarff 

from livelossplot import PlotLosses
from collections import Counter

from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

import tqdm as tqdm_normal

from IPython.display import Image
from IPython.display import display, clear_output

import pandas as pd

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
#tf.get_logger().setLevel('WARNING')
#tf.autograph.set_verbosity(1)

import tensorflow_addons as tfa
#import tensorflow_probability as tfp
#from tensorflow_probability.stats import expected_calibration_error

from keras import backend as K

import seaborn as sns
sns.set_style("darkgrid")

import time
import random

from utilities.utilities_GRANDE import *
from utilities.GRANDE import *
from utilities.TabSurvey.models.node import NODE

from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

import interruptingcow
from interruptingcow import timeout
import signal

from joblib import Parallel, delayed

from itertools import product
from collections.abc import Iterable
import collections

from copy import deepcopy
import timeit

import functools
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import scipy

from pathlib import Path
import csv
import dill

import joblib
import ray
from ray.util.joblib import register_ray
#register_ray()
#ray.init(logging_level=logging.ERROR)

from tabulate import tabulate

#from pydl85 import DL85Classifier

import numbers
import shap
from datetime import datetime

#from utilities.node_tf.networks.model import NODE
from utilities.TabSurvey.utils.load_data import load_data
import yaml
import json

import optuna
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from ray.tune import JupyterNotebookReporter

import openml

import contextlib

from functools import reduce

import re

def string_to_list(s):
    # Extract the list content inside the brackets, accounting for potential white spaces
    if 'ListWrapper' in s:
        
        match = re.search(r'ListWrapper\(\[(.*)\]\)', s)
        
        # If there's no match, raise an error or return a default value
        if not match:
            raise ValueError(f"String '{s}' does not match the expected format")
        
        content = match.group(1).strip()
        
        # If content is empty return an empty list
        if not content:
            return []
    else:
        match = re.search(r'\[(.*)\]', s)
        
        # If there's no match, raise an error or return a default value
        if not match:
            raise ValueError(f"String '{s}' does not match the expected format")
        
        content = match.group(1).strip()
        
        # If content is empty return an empty list
        if not content:
            return []
    # Convert the string of numbers to a list of integers
    return [int(x) for x in content.split(',')]


def free_memory(sleep_time=0.1):
    """Black magic function to free torch memory and some jupyter whims."""
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)


def AMS_metric(y_true, y_pred):
    
    from sklearn.metrics import confusion_matrix

    def AMS(s, b):
        """ Approximate Median Significance defined as:
            AMS = sqrt(
                    2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
                  )        
        where b_r = 10, b = background, s = signal, log is natural logarithm """

        br = 10.0
        radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
        if radicand < 0:
            print('radicand is negative. Exiting')
            exit()
        else:
            return math.sqrt(radicand)
        
    ((tn, fp), (fn, tp)) = confusion_matrix(y_true, np.round(y_pred))
    
    tpr = (tp /(tp + fn))
    sp = (tn / (fp + tn))
    fpr = 1 - sp

    score = AMS(tpr, fpr)    
    
    return score


def flatten_list(l):
    
    def flatten(l):
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el
                
    flat_l = flatten(l)
    
    return list(flat_l)

def flatten_list_one_level(l):
    result = []
    for sublist in l:
        if isinstance(sublist, list):
            result.extend(sublist)
        else:
            result.append(sublist)
    return result



def flatten_dict(d, parent_key='', sep='__'):

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def mergeDict(dict1, dict2):
    #Merge dictionaries and keep values of common keys in list
    newDict = {**dict1, **dict2}
    for key, value in newDict.items():
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                newDict[key] = mergeDict(dict1[key], value)
            elif isinstance(dict1[key], list) and isinstance(value, list):
                newDict[key] = dict1[key]
                newDict[key].extend(value)
            elif isinstance(dict1[key], list) and not isinstance(value, list):
                newDict[key] = dict1[key]
                newDict[key].extend([value])
            elif not isinstance(dict1[key], list) and isinstance(value, list):
                newDict[key] = [dict1[key]]
                newDict[key].extend(value)
            else:
                newDict[key] = [dict1[key], value]
    return newDict


def normalize_data(X_data, normalizer_list=None, technique='min-max', low=-1, high=1, quantile_noise=1e-3, random_state=42, exclude_columns=[]):
    if normalizer_list is None:
        normalizer_list = []
        if isinstance(X_data, pd.DataFrame):
            if technique != 'quantile':
            
                for column_name in X_data:
                    if column_name not in exclude_columns:
                        if technique == 'min-max':
                            scaler = MinMaxScaler(feature_range=(low, high))
                        elif technique == 'mean':
                            scaler = StandardScaler()

                        scaler.fit(X_data[column_name].values.reshape(-1, 1))

                        X_data[column_name] = scaler.transform(X_data[column_name].values.reshape(-1, 1)).ravel()
                        normalizer_list.append(scaler)
                
            else:
                
                Z_data = X_data.drop(exclude_columns, axis=1)
                relevant_columns = Z_data.columns
                
                if len(relevant_columns) > 0:

                    quantile_train = np.copy(Z_data.values).astype(np.float64)
                    if quantile_noise > 0:
                        np.random.seed(random_state)
                        stds = np.std(quantile_train, axis=0, keepdims=True)
                        noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                        quantile_train += noise_std * np.random.randn(*quantile_train.shape)                        

                    scaler = QuantileTransformer(output_distribution='normal', random_state=random_state)
                    scaler.fit(quantile_train)   
                    Z_data = pd.DataFrame(scaler.transform(Z_data.values), columns=Z_data.columns, index=Z_data.index)

                    X_data = pd.concat([X_data[exclude_columns], pd.DataFrame(Z_data, index=X_data.index, columns=relevant_columns)], axis=1)

                    normalizer_list.append(scaler) 
            

    else:
        if isinstance(X_data, pd.DataFrame):
            if technique != 'quantile':
                for column_name, scaler in zip(X_data, normalizer_list):
                    X_data[column_name] = scaler.transform(X_data[column_name].values.reshape(-1, 1)).ravel()
            else:
                Z_data = X_data.drop(exclude_columns, axis=1)
                relevant_columns = Z_data.columns
                
                if len(relevant_columns) > 0:
                    Z_data = pd.DataFrame(normalizer_list[0].transform(Z_data.values), columns=Z_data.columns, index=Z_data.index)

                    X_data = pd.concat([X_data[exclude_columns], pd.DataFrame(Z_data, index=X_data.index, columns=relevant_columns)], axis=1)
                
    return X_data, normalizer_list


def rebalance_data(X_train, 
                   y_train,
                   balance_ratio=0.25, 
                   strategy='SMOTE',#'SMOTE', 
                   seed=42, 
                   verbosity=0):#, strategy='SMOTE'
    
    if balance_ratio > 0:
        min_label = min(Counter(y_train).values())
        sum_label = sum(Counter(y_train).values())
    
        min_ratio = min_label/sum_label
        if verbosity > 0:
            print('Min Ratio: ', str(min_ratio))    
        if min_ratio <= balance_ratio/(len(Counter(y_train).values()) - 1):
            from imblearn.over_sampling import RandomOverSampler, SMOTEN#, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, SMOTENC
            try:
                if strategy == 'SMOTE':
                    oversample = SMOTEN(k_neighbors=sklearn.neighbors.NearestNeighbors(n_jobs=1))
                else:
                    oversample = RandomOverSampler(sampling_strategy='auto', random_state=seed)
    
                X_train, y_train = oversample.fit_resample(X_train, y_train)
            except ValueError:
                X_train, y_train = oversample.fit_resample(X_train, y_train)
                
            min_label = min(Counter(y_train).values())
            sum_label = sum(Counter(y_train).values())
            min_ratio = min_label/sum_label
            if verbosity > 0:
                print('Min Ratio: ', str(min_ratio))    

    return X_train, y_train



def calculate_class_weights(y_data):
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_data), y = y_data)
    #class_weights = class_weights/sum(class_weights)#
    return class_weights


def calculate_sample_weights(y_data):
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_data), y = y_data)
    #class_weights = class_weights/sum(class_weights)#    
    sample_weights = sklearn.utils.class_weight.compute_sample_weight(class_weight = 'balanced', y =y_data)
    #sample_weights = sample_weights/sum(class_weights)
    return sample_weights


def preprocess_data(X_data, 
                    y_data,
                    categorical_indicator,
                    config,
                    random_seed=42,
                    verbosity=0,
                    hpo=False):
    
    number_of_classes = len(np.unique(y_data.values)) if config['GRANDE']['objective'] == 'classification' else 1
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    tf.keras.utils.set_random_seed(random_seed)    
    
    if verbosity > 0:
        print('Original Data Shape (selected): ', X_data.shape)

    total_samples = X_data.shape[0]
    
    if config['preprocessing']['encoding_type'] is not None:
        numerical_threshold = min(100, total_samples//4)
    else:
        numerical_threshold = 1
    one_hot_encode_threshold = config['preprocessing']['one_hot_encode_threshold']#10
    
    binary_features = []
    categorical_features_low_cardinality = []
    categorical_features_high_cardinality = []
    numerical_features = []
    
    categorical_feature_indices = []
    for index, column_name in enumerate(X_data.columns):
        #num_categories = len(np.unique(X_data[column_name].values))
        num_categories = len(X_data[column_name].value_counts())
        try:
            is_categorical = categorical_indicator[index]
            if is_categorical:
                if num_categories <= 2:
                    binary_features.append(column_name)
                elif num_categories < one_hot_encode_threshold:
                    categorical_features_low_cardinality.append(column_name)
                else:# num_categories >= one_hot_encode_threshold and num_categories < numerical_threshold:
                    categorical_features_high_cardinality.append(column_name)
            else:
                #if X_data[column_name].values.dtype == 'O':
                #    categorical_features_high_cardinality.append(column_name)
                #else:
                #    numerical_features.append(column_name)
                try: 
                    X_data[column_name] = X_data[column_name].astype(float)
                except:
                    encoder = ce.OrdinalEncoder(cols=[column_name])
                    X_data = encoder.fit_transform(X_data)              
                numerical_features.append(column_name)   
        except TypeError as e: #no categorical_indicator defined
            if num_categories < numerical_threshold:
                if num_categories <= 2:
                    binary_features.append(column_name)
                elif num_categories < one_hot_encode_threshold:
                    categorical_features_low_cardinality.append(column_name)
                else:# num_categories >= one_hot_encode_threshold and num_categories < numerical_threshold:
                    categorical_features_high_cardinality.append(column_name)
            else:
                #if X_data[column_name].values.dtype == 'O':
                #    categorical_features_high_cardinality.append(column_name)
                #else:
                #    numerical_features.append(column_name)
                try: 
                    X_data[column_name] = X_data[column_name].astype(float)
                except:
                    encoder = ce.OrdinalEncoder(cols=[column_name])
                    X_data = encoder.fit_transform(X_data)              
                numerical_features.append(column_name)              
           
    #print('categorical_features_low_cardinality', categorical_features_low_cardinality)
    #print('categorical_features_high_cardinality', categorical_features_high_cardinality)
    #print('numerical_features', numerical_features)
    if verbosity > 0:
        print('Original Data Shape (encoded): ', X_data.shape)
        if config['GRANDE']['objective'] == 'classification':
            print('Original Data Class Distribution: ', y_data[y_data>=0.5].shape[0], ' (true) /', y_data[y_data<0.5].shape[0], ' (false)')   
        
    if False: 
        X_data_raw = X_data.copy(deep=True)
        for column_name in X_data_raw.columns:
            try: 
                placeholder = X_data_raw[column_name].astype(float)
            except:
                encoder = ce.OrdinalEncoder(cols=[column_name])
                X_data_raw = encoder.fit_transform(X_data_raw)     
    else:
        categorical_feature_indices = []
        X_data_raw = X_data.copy(deep=True)
        for col in X_data_raw.select_dtypes(include=['object']):
            X_data_raw[col] = X_data_raw[col].astype('category')                    
        for index, column_name in enumerate(X_data_raw.columns):
            if not ('float' in str(X_data_raw[column_name].dtype) or 'int' in str(X_data_raw[column_name].dtype)):
            #if column_name not in numerical_features:   
                if X_data_raw[column_name].dtype == 'category':
                    X_data_raw[column_name] = X_data_raw[column_name].cat.codes
                    if X_data_raw[column_name].min() < 0:
                        X_data_raw[column_name] = X_data_raw[column_name] + 1
                else:
                    encoder = ce.OrdinalEncoder(cols=[column_name])
                    X_data_raw = encoder.fit_transform(X_data_raw)
                X_data_raw[column_name] = X_data_raw[column_name].astype('category') 
                categorical_feature_indices.append(index)
        
            if np.max(X_data_raw[column_name].to_numpy()) >= len(X_data[column_name].value_counts()) and not column_name in numerical_features:
                encoder = ce.OrdinalEncoder(cols=[column_name])
                X_data_raw = encoder.fit_transform(X_data_raw) 
                X_data_raw[column_name] = X_data_raw[column_name].astype('category') 
                if index not in categorical_feature_indices:
                    categorical_feature_indices.append(index)

                
        
    #categorical_feature_indices = []
    #for index, column in enumerate(X_train_raw_list[0].columns):
    #    if column in numerical_features:
    #        pass
    #    else:
    #        categorical_feature_indices.append(index)                 

    encoder = LabelEncoder()
    for column in binary_features:
        encoder = encoder.fit(X_data[column].values)
        X_data[column] = encoder.transform(X_data[column].values)      
    
    if config['preprocessing']['encoding_type'] is not None:
        encoder = ce.OneHotEncoder(cols=categorical_features_low_cardinality)
        X_data = encoder.fit_transform(X_data)
    else:
        encoder = ce.OrdinalEncoder(cols=categorical_features_low_cardinality)
        X_data = encoder.fit_transform(X_data)
        
    if config['computation']['cv_num_eval'] == 1:
        stratify = y_data if config['GRANDE']['objective'] == 'classification' else None
        cv_generator = [train_test_split(X_data.index, test_size=0.2, random_state=random_seed, stratify=stratify)]
    else:
        if config['GRANDE']['objective'] == 'regression':
            kf = KFold(n_splits=config['computation']['cv_num_eval'], shuffle=True, random_state=random_seed)
        else:
            kf = StratifiedKFold(n_splits=config['computation']['cv_num_eval'], shuffle=True, random_state=random_seed)   
        cv_generator = kf.split(X_data, y_data)

    X_train_list = []
    y_train_list = []
    X_valid_list = []
    y_valid_list = []
    
    X_train_no_valid_list = []
    y_train_no_valid_list = []
    
    X_test_list = []
    y_test_list = []
    
    X_test_no_valid_list = []
    y_test_no_valid_list = []    
    normalizer_list_list = []
    
    X_train_raw_list = []
    y_train_raw_list = []
    X_valid_raw_list = []
    y_valid_raw_list = []
    
    X_train_raw_no_valid_list = []
    y_train_raw_no_valid_list = []
    
    X_test_raw_list = []
    y_test_raw_list = []
    
    X_test_raw_no_valid_list = []
    y_test_raw_no_valid_list = []  
    
    X_train_list_cv_list = []
    y_train_list_cv_list = []
    X_train_no_valid_list_cv_list = []
    y_train_no_valid_list_cv_list = []    
    X_valid_list_cv_list = []
    y_valid_list_cv_list = []
    X_test_list_cv_list = []
    y_test_list_cv_list = []
    X_test_no_valid_list_cv_list = []
    y_test_no_valid_list_cv_list = []    
    normalizer_list_list_cv_list = []   
    
    X_train_raw_list_cv_list = []
    y_train_raw_list_cv_list = []
    X_train_raw_no_valid_list_cv_list = []
    y_train_raw_no_valid_list_cv_list = []    
    X_valid_raw_list_cv_list = []
    y_valid_raw_list_cv_list = []
    X_test_raw_list_cv_list = []
    y_test_raw_list_cv_list = []
    X_test_raw_no_valid_list_cv_list = []
    y_test_raw_no_valid_list_cv_list = []   
    
    exclude_columns = []
    
    for i, (train_index_with_valid, test_index) in enumerate(cv_generator):
        
        if hpo and config['computation']['cv_num_hpo'] >= 1:
            X_train_list_cv = []
            y_train_list_cv = []
            
            X_train_no_valid_list_cv = []
            y_train_no_valid_list_cv = []  
            
            X_valid_list_cv = []
            y_valid_list_cv = []           
            
            X_test_list_cv = []
            y_test_list_cv = []
            
            X_test_no_valid_list_cv = []
            y_test_no_valid_list_cv = []
            
            normalizer_list_list_cv = []
            
            X_train_raw_list_cv = []
            y_train_raw_list_cv = []
            
            X_train_raw_no_valid_list_cv = []
            y_train_raw_no_valid_list_cv = []  
            
            X_valid_raw_list_cv = []
            y_valid_raw_list_cv = []           
            
            X_test_raw_list_cv = []
            y_test_raw_list_cv = []
            
            X_test_raw_no_valid_list_cv = []
            y_test_raw_no_valid_list_cv = []            
            
            X_test = X_data.iloc[test_index]
            y_test = y_data.iloc[test_index]         
            
            X_data_cv = X_data.iloc[train_index_with_valid]
            y_data_cv = y_data.iloc[train_index_with_valid]
            
            X_test_raw = X_data_raw.iloc[test_index]
            y_test_raw = y_data.iloc[test_index]         
            
            X_data_raw_cv = X_data_raw.iloc[train_index_with_valid]
            y_data_raw_cv = y_data.iloc[train_index_with_valid]            

            if config['computation']['cv_num_hpo'] == 1:
                stratify = y_data.iloc[train_index_with_valid] if config['GRANDE']['objective'] == 'classification' else None
                cv_generator = [train_test_split(train_index_with_valid, test_size=test_size, random_state=random_seed, stratify=stratify)]
            else:
                if config['GRANDE']['objective'] == 'regression':
                    kf = KFold(n_splits=config['computation']['cv_num_hpo'], shuffle=True, random_state=random_seed)
                    cv_generator = kf.split(X_data_cv, y_data_cv)
                else:
                    kf = StratifiedKFold(n_splits=config['computation']['cv_num_hpo'], shuffle=True, random_state=random_seed)   
                    cv_generator = kf.split(X_data_cv, y_data_cv)

            
            for j, (train_index_cv, test_index_cv) in enumerate(cv_generator):
                      
                X_train_no_valid_cv = X_data_cv.iloc[train_index_cv]
                X_test_no_valid_cv = X_data_cv.iloc[test_index_cv]

                y_train_no_valid_cv = y_data_cv.iloc[train_index_cv]
                y_test_no_valid_cv = y_data_cv.iloc[test_index_cv]

                X_train_raw_no_valid_cv = X_data_raw_cv.iloc[train_index_cv]
                y_train_raw_no_valid_cv = y_data_raw_cv.iloc[train_index_cv]

                X_test_raw_no_valid_cv = X_data_raw_cv.iloc[test_index_cv]
                y_test_raw_no_valid_cv = y_data_raw_cv.iloc[test_index_cv]

                if len(categorical_features_high_cardinality) > 0:
                    if config['preprocessing']['encoding_type'] == 'LOO': #LOO
                        encoder = ce.LeaveOneOutEncoder(cols=categorical_features_high_cardinality)

                        #encoder = encoder.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
                        encoder = encoder.fit(X_train_no_valid_cv, y_train_no_valid_cv)
                        X_train_no_valid_cv = encoder.transform(X_train_no_valid_cv)
                        X_test_no_valid_cv = encoder.transform(X_test_no_valid_cv)
                    elif config['preprocessing']['encoding_type'] == 'GLMM': #LOO
                        encoder = ce.GLMMEncoder(cols=categorical_features_high_cardinality)

                        encoder = encoder.fit(X_train_no_valid_cv, y_train_no_valid_cv)
                        X_train_no_valid_cv = encoder.transform(X_train_no_valid_cv)
                        X_test_no_valid_cv = encoder.transform(X_test_no_valid_cv)
                    elif config['preprocessing']['encoding_type'] == 'BackwardDifference': #LOO
                        encoder = ce.BackwardDifferenceEncoder(cols=categorical_features_high_cardinality)

                        encoder = encoder.fit(X_train_no_valid_cv, y_train_no_valid_cv)
                        X_train_no_valid_cv = encoder.transform(X_train_no_valid_cv)
                        X_test_no_valid_cv = encoder.transform(X_test_no_valid_cv)                         
                        
                    else:
                        encoder = ce.OrdinalEncoder(cols=categorical_features_high_cardinality)

                        #encoder = encoder.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
                        encoder = encoder.fit(X_train_no_valid_cv, y_train_no_valid_cv)
                        X_train_no_valid_cv = encoder.transform(X_train_no_valid_cv)
                        X_test_no_valid_cv = encoder.transform(X_test_no_valid_cv)                    
                        

                if config['preprocessing']['normalization_technique'] is not None:
                    #_, normalizer_list = normalize_data(pd.concat([X_train, X_valid]), technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
                    #X_train, _ = normalize_data(X_train, normalizer_list=normalizer_list, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)

                    X_train_no_valid_cv, normalizer_list_cv = normalize_data(X_train_no_valid_cv, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
                    X_test_no_valid_cv, _ = normalize_data(X_test_no_valid_cv, normalizer_list=normalizer_list_cv, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
                else:
                    normalizer_list_cv = None            

                #X_train = X_train.astype(np.float64)        
                #X_valid = X_valid.astype(np.float64)        
                #X_test = X_test.astype(np.float64)        

                if config['GRANDE']['objective'] == 'classification' and number_of_classes == 2:
                    X_train_no_valid_cv, y_train_no_valid_cv = rebalance_data(X_train_no_valid_cv, 
                                                                      y_train_no_valid_cv,
                                                                      balance_ratio=config['preprocessing']['balance_threshold'],  
                                                                      strategy='SMOTE',
                                                                      verbosity=verbosity)  
                    
                    X_train_raw_no_valid_cv, y_train_raw_no_valid_cv = rebalance_data(X_train_raw_no_valid_cv, 
                                                                      y_train_raw_no_valid_cv,
                                                                      balance_ratio=config['preprocessing']['balance_threshold'],  
                                                                      strategy='SMOTE',
                                                                      verbosity=verbosity)  
                    
                X_train_no_valid_list_cv.append(X_train_no_valid_cv)
                X_test_no_valid_list_cv.append(X_test_no_valid_cv)

                y_train_no_valid_list_cv.append(y_train_no_valid_cv)
                y_test_no_valid_list_cv.append(y_test_no_valid_cv)

                normalizer_list_list_cv.append(normalizer_list_cv)   
                
                X_train_raw_no_valid_list_cv.append(X_train_raw_no_valid_cv)
                X_test_raw_no_valid_list_cv.append(X_test_raw_no_valid_cv)

                y_train_raw_no_valid_list_cv.append(y_train_raw_no_valid_cv)
                y_test_raw_no_valid_list_cv.append(y_test_raw_no_valid_cv)                
            
            if config['computation']['cv_num_hpo'] == 1:
                stratify = y_data.iloc[train_index_with_valid] if config['GRANDE']['objective'] == 'classification' else None
                cv_generator = [train_test_split(train_index_with_valid, test_size=test_size, random_state=random_seed, stratify=stratify)]
            else:
                if config['GRANDE']['objective'] == 'regression':
                    kf = KFold(n_splits=config['computation']['cv_num_hpo'], shuffle=True, random_state=random_seed)
                    cv_generator = kf.split(X_data_cv, y_data_cv)
                else:
                    kf = StratifiedKFold(n_splits=config['computation']['cv_num_hpo'], shuffle=True, random_state=random_seed)   
                    cv_generator = kf.split(X_data_cv, y_data_cv)

            
            for j, (train_index_with_valid_cv, test_index_cv) in enumerate(cv_generator):
                stratify = y_data_cv.iloc[train_index_with_valid_cv] if config['GRANDE']['objective'] == 'classification' else None
                train_index_cv, valid_index_cv = train_test_split(train_index_with_valid_cv, test_size=0.1, random_state=random_seed, stratify=stratify)
                
                #train_index_cv = train_index_with_valid_cv[train_index_cv]
                #valid_index_cv = train_index_with_valid_cv[valid_index_cv]
                
                ###print(X_data_cv.shape) 
                ###print(len(train_index_cv), max(train_index_cv))
                ###print(sorted(train_index_cv)[:10])
                ###print(len(valid_index_cv), max(valid_index_cv))
                ###print(sorted(valid_index_cv)[:10])
                      
                X_train_cv = X_data_cv.iloc[train_index_cv]
                X_valid_cv = X_data_cv.iloc[valid_index_cv]
                X_test_cv = X_data_cv.iloc[test_index_cv]

                y_train_cv = y_data_cv.iloc[train_index_cv]
                y_valid_cv = y_data_cv.iloc[valid_index_cv]
                y_test_cv = y_data_cv.iloc[test_index_cv]
                     
                X_train_raw_cv = X_data_raw_cv.iloc[train_index_cv]
                X_valid_raw_cv = X_data_raw_cv.iloc[valid_index_cv]
                X_test_raw_cv =  X_data_raw_cv.iloc[test_index_cv]

                y_train_raw_cv = y_data_raw_cv.iloc[train_index_cv]
                y_valid_raw_cv = y_data_raw_cv.iloc[valid_index_cv]
                y_test_raw_cv = y_data_raw_cv.iloc[test_index_cv]

                if len(categorical_features_high_cardinality) > 0:
                    if config['preprocessing']['encoding_type'] == 'LOO': #LOO
                        encoder = ce.LeaveOneOutEncoder(cols=categorical_features_high_cardinality)

                        #encoder = encoder.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
                        encoder = encoder.fit(X_train_cv, y_train_cv)
                        X_train_cv = encoder.transform(X_train_cv)
                        X_valid_cv = encoder.transform(X_valid_cv)
                        X_test_cv = encoder.transform(X_test_cv)
                    elif config['preprocessing']['encoding_type'] == 'GLMM': #LOO
                        encoder = ce.GLMMEncoder(cols=categorical_features_high_cardinality)

                        encoder = encoder.fit(X_train_cv, y_train_cv)
                        X_train_cv = encoder.transform(X_train_cv)
                        X_valid_cv = encoder.transform(X_valid_cv)
                        X_test_cv = encoder.transform(X_test_cv)  
                    elif config['preprocessing']['encoding_type'] == 'BackwardDifference': #LOO
                        encoder = ce.BackwardDifferenceEncoder(cols=categorical_features_high_cardinality)

                        encoder = encoder.fit(X_train_cv, y_train_cv)
                        X_train_cv = encoder.transform(X_train_cv)
                        X_valid_cv = encoder.transform(X_valid_cv)
                        X_test_cv = encoder.transform(X_test_cv)                    
                        
                    else:
                        encoder = ce.OrdinalEncoder(cols=categorical_features_high_cardinality)

                        #encoder = encoder.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
                        encoder = encoder.fit(X_train_cv, y_train_cv)
                        X_train_cv = encoder.transform(X_train_cv)
                        X_valid_cv = encoder.transform(X_valid_cv)
                        X_test_cv = encoder.transform(X_test_cv)                  
                        
                        

                if config['preprocessing']['normalization_technique'] is not None:
                    #_, normalizer_list = normalize_data(pd.concat([X_train, X_valid]), technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
                    #X_train, _ = normalize_data(X_train, normalizer_list=normalizer_list, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)

                    X_train_cv, normalizer_list_cv = normalize_data(X_train_cv, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
                    X_valid_cv, _ = normalize_data(X_valid_cv, normalizer_list=normalizer_list_cv, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
                    X_test_cv, _ = normalize_data(X_test_cv, normalizer_list=normalizer_list_cv, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
                else:
                    normalizer_list_cv = None            

                #X_train = X_train.astype(np.float64)        
                #X_valid = X_valid.astype(np.float64)        
                #X_test = X_test.astype(np.float64)        

                if config['GRANDE']['objective'] == 'classification' and number_of_classes == 2:
                    X_train_cv, y_train_cv = rebalance_data(X_train_cv, 
                                                              y_train_cv,
                                                              balance_ratio=config['preprocessing']['balance_threshold'],  
                                                              strategy='SMOTE',
                                                              verbosity=verbosity)  
                    X_train_raw_cv, y_train_raw_cv = rebalance_data(X_train_raw_cv, 
                                                              y_train_raw_cv,
                                                              balance_ratio=config['preprocessing']['balance_threshold'],  
                                                              strategy='SMOTE',
                                                              verbosity=verbosity)                    

                X_train_list_cv.append(X_train_cv)
                X_valid_list_cv.append(X_valid_cv)
                X_test_list_cv.append(X_test_cv)

                y_train_list_cv.append(y_train_cv)
                y_valid_list_cv.append(y_valid_cv)
                y_test_list_cv.append(y_test_cv)

                normalizer_list_list_cv.append(normalizer_list_cv)   
                
                X_train_raw_list_cv.append(X_train_raw_cv)
                X_valid_raw_list_cv.append(X_valid_raw_cv)
                X_test_raw_list_cv.append(X_test_raw_cv)

                y_train_raw_list_cv.append(y_train_raw_cv)
                y_valid_raw_list_cv.append(y_valid_raw_cv)
                y_test_raw_list_cv.append(y_test_raw_cv)
                

            X_train_list_cv_list.append(X_train_list_cv)
            y_train_list_cv_list.append(y_train_list_cv)
            X_train_no_valid_list_cv_list.append(X_train_no_valid_list_cv)
            y_train_no_valid_list_cv_list.append(y_train_no_valid_list_cv)            
            X_valid_list_cv_list.append(X_valid_list_cv)
            y_valid_list_cv_list.append(y_valid_list_cv)
            X_test_list_cv_list.append(X_test_list_cv)
            y_test_list_cv_list.append(y_test_list_cv)
            X_test_no_valid_list_cv_list.append(X_test_no_valid_list_cv)
            y_test_no_valid_list_cv_list.append(y_test_no_valid_list_cv)            
            normalizer_list_list_cv_list.append(normalizer_list_list_cv) 
        
            X_train_raw_list_cv_list.append(X_train_raw_list_cv)
            y_train_raw_list_cv_list.append(y_train_raw_list_cv)
            X_train_raw_no_valid_list_cv_list.append(X_train_raw_no_valid_list_cv)
            y_train_raw_no_valid_list_cv_list.append(y_train_raw_no_valid_list_cv)            
            X_valid_raw_list_cv_list.append(X_valid_raw_list_cv)
            y_valid_raw_list_cv_list.append(y_valid_raw_list_cv)
            X_test_raw_list_cv_list.append(X_test_raw_list_cv)
            y_test_raw_list_cv_list.append(y_test_raw_list_cv)
            X_test_raw_no_valid_list_cv_list.append(X_test_raw_no_valid_list_cv)
            y_test_raw_no_valid_list_cv_list.append(y_test_raw_no_valid_list_cv)           
        
        if config['computation']['cv_num_hpo'] > 0:
            stratify = y_data.iloc[train_index_with_valid] if config['GRANDE']['objective'] == 'classification' else None
            train_index, valid_index = train_test_split(train_index_with_valid, test_size=0.1, random_state=random_seed, stratify=stratify)
            X_train = X_data.iloc[train_index]
            X_valid = X_data.iloc[valid_index]
            X_test = X_data.iloc[test_index]

            y_train = y_data.iloc[train_index]
            y_valid = y_data.iloc[valid_index]
            y_test = y_data.iloc[test_index]            

            X_train_no_valid = X_data.iloc[train_index_with_valid]
            y_train_no_valid = y_data.iloc[train_index_with_valid]

            X_test_no_valid = X_data.iloc[test_index]
            y_test_no_valid = y_data.iloc[test_index]
            
            
            X_train_raw = X_data_raw.iloc[train_index]
            X_valid_raw = X_data_raw.iloc[valid_index]
            X_test_raw = X_data_raw.iloc[test_index]

            y_train_raw = y_data.iloc[train_index]
            y_valid_raw = y_data.iloc[valid_index]
            y_test_raw = y_data.iloc[test_index]

            X_train_raw_no_valid = X_data_raw.iloc[train_index_with_valid]
            y_train_raw_no_valid = y_data.iloc[train_index_with_valid]

            X_test_raw_no_valid = X_data_raw.iloc[test_index]
            y_test_raw_no_valid = y_data.iloc[test_index]
        else:
            X_train = X_data.iloc[train_index_with_valid]
            X_valid = X_data.iloc[test_index]
            X_test = X_data.iloc[test_index]   
                    
            y_train = y_data.iloc[train_index_with_valid]
            y_valid = y_data.iloc[test_index]
            y_test = y_data.iloc[test_index]            

            X_train_no_valid = X_data.iloc[train_index_with_valid]
            y_train_no_valid = y_data.iloc[train_index_with_valid]

            X_test_no_valid = X_data.iloc[test_index]
            y_test_no_valid = y_data.iloc[test_index]
            
                    
            X_train_raw = X_data_raw.iloc[train_index_with_valid]
            X_valid_raw = X_data_raw.iloc[test_index]
            X_test_raw = X_data_raw.iloc[test_index]

            y_train_raw = y_data.iloc[train_index_with_valid]
            y_valid_raw = y_data.iloc[test_index]
            y_test_raw = y_data.iloc[test_index]

            X_train_raw_no_valid = X_data_raw.iloc[train_index_with_valid]
            y_train_raw_no_valid = y_data.iloc[train_index_with_valid]

            X_test_raw_no_valid = X_data_raw.iloc[test_index]
            y_test_raw_no_valid = y_data.iloc[test_index]              

            
        if len(categorical_features_high_cardinality) > 0:
            if config['preprocessing']['encoding_type'] == 'LOO': #LOO
                encoder = ce.LeaveOneOutEncoder(cols=categorical_features_high_cardinality)

                #encoder = encoder.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
                encoder = encoder.fit(X_train_no_valid, y_train_no_valid)
                X_train_no_valid = encoder.transform(X_train_no_valid)
                X_test_no_valid = encoder.transform(X_test_no_valid)
                
            elif config['preprocessing']['encoding_type'] == 'GLMM': #LOO
                encoder = ce.GLMMEncoder(cols=categorical_features_high_cardinality)

                #encoder = encoder.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
                encoder = encoder.fit(X_train_no_valid, y_train_no_valid)
                X_train_no_valid = encoder.transform(X_train_no_valid)
                X_test_no_valid = encoder.transform(X_test_no_valid)   
            elif config['preprocessing']['encoding_type'] == 'BackwardDifference': #LOO
                encoder = ce.BackwardDifferenceEncoder(cols=categorical_features_high_cardinality)

                #encoder = encoder.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
                encoder = encoder.fit(X_train_no_valid, y_train_no_valid)
                X_train_no_valid = encoder.transform(X_train_no_valid)
                X_test_no_valid = encoder.transform(X_test_no_valid)                   
            else:
                encoder = ce.OrdinalEncoder(cols=categorical_features_high_cardinality)

                #encoder = encoder.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
                encoder = encoder.fit(X_train_no_valid, y_train_no_valid)
                X_train_no_valid = encoder.transform(X_train_no_valid)
                X_test_no_valid = encoder.transform(X_test_no_valid)                
            del encoder
        if config['preprocessing']['normalization_technique'] is not None:
            #_, normalizer_list = normalize_data(pd.concat([X_train, X_valid]), technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
            #X_train, _ = normalize_data(X_train, normalizer_list=normalizer_list, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)

            X_train_no_valid, normalizer_list = normalize_data(X_train_no_valid, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
            X_test_no_valid, _ = normalize_data(X_test_no_valid, normalizer_list=normalizer_list, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
        else:
            normalizer_list_cv = None            

        #X_train = X_train.astype(np.float64)        
        #X_valid = X_valid.astype(np.float64)        
        #X_test = X_test.astype(np.float64)        

        if config['GRANDE']['objective'] == 'classification' and number_of_classes == 2:
            X_train_no_valid, y_train_no_valid = rebalance_data(X_train_no_valid, 
                                                                          y_train_no_valid,
                                                                          balance_ratio=config['preprocessing']['balance_threshold'],  
                                                                          strategy='SMOTE',
                                                                          verbosity=verbosity)   
            
            X_train_raw_no_valid, y_train_raw_no_valid = rebalance_data(X_train_raw_no_valid, 
                                                                          y_train_raw_no_valid,
                                                                          balance_ratio=config['preprocessing']['balance_threshold'],  
                                                                          strategy='SMOTE',
                                                                          verbosity=verbosity)             

        if len(categorical_features_high_cardinality) > 0:
            if config['preprocessing']['encoding_type'] == 'LOO': #LOO
                encoder = ce.LeaveOneOutEncoder(cols=categorical_features_high_cardinality)

                #encoder = encoder.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
                encoder = encoder.fit(X_train, y_train)
                X_train = encoder.transform(X_train)
                X_valid = encoder.transform(X_valid)
                X_test = encoder.transform(X_test)
            elif config['preprocessing']['encoding_type'] == 'GLMM': #LOO
                encoder = ce.GLMMEncoder(cols=categorical_features_high_cardinality)

                encoder = encoder.fit(X_train, y_train)
                X_train = encoder.transform(X_train)
                X_valid = encoder.transform(X_valid)
                X_test = encoder.transform(X_test)
            elif config['preprocessing']['encoding_type'] == 'BackwardDifference': #LOO
                encoder = ce.BackwardDifferenceEncoder(cols=categorical_features_high_cardinality)

                encoder = encoder.fit(X_train, y_train)
                X_train = encoder.transform(X_train)
                X_valid = encoder.transform(X_valid)
                X_test = encoder.transform(X_test)
            else:
                encoder = ce.OrdinalEncoder(cols=categorical_features_high_cardinality)

                #encoder = encoder.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
                encoder = encoder.fit(X_train, y_train)
                X_train = encoder.transform(X_train)
                X_valid = encoder.transform(X_valid)
                X_test = encoder.transform(X_test)
            del encoder
                
        if config['preprocessing']['normalization_technique'] is not None:
            #_, normalizer_list = normalize_data(pd.concat([X_train, X_valid]), technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
            #X_train, _ = normalize_data(X_train, normalizer_list=normalizer_list, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)

            X_train, normalizer_list = normalize_data(X_train, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
            X_valid, _ = normalize_data(X_valid, normalizer_list=normalizer_list, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
            X_test, _ = normalize_data(X_test, normalizer_list=normalizer_list, technique=config['preprocessing']['normalization_technique'], quantile_noise=config['preprocessing']['quantile_noise'], exclude_columns=exclude_columns)
        else:
            normalizer_list = None            

        #X_train = X_train.astype(np.float64)        
        #X_valid = X_valid.astype(np.float64)        
        #X_test = X_test.astype(np.float64)        
        if config['GRANDE']['objective'] == 'classification' and number_of_classes == 2:
            X_train, y_train = rebalance_data(X_train, 
                                              y_train,
                                              balance_ratio=config['preprocessing']['balance_threshold'],  
                                              strategy='SMOTE',
                                              verbosity=verbosity)  
            
            X_train_raw, y_train_raw = rebalance_data(X_train_raw, 
                                              y_train_raw,
                                              balance_ratio=config['preprocessing']['balance_threshold'],  
                                              strategy='SMOTE',
                                              verbosity=verbosity)             
        X_train_list.append(X_train)
        X_train_no_valid_list.append(X_train_no_valid)
        X_valid_list.append(X_valid)
        X_test_list.append(X_test)
        X_test_no_valid_list.append(X_test_no_valid)

        y_train_list.append(y_train)
        y_train_no_valid_list.append(y_train_no_valid)
        y_valid_list.append(y_valid)
        y_test_list.append(y_test)
        y_test_no_valid_list.append(y_test_no_valid)

        normalizer_list_list.append(normalizer_list)
        
        X_train_raw_list.append(X_train_raw)
        X_train_raw_no_valid_list.append(X_train_raw_no_valid)
        X_valid_raw_list.append(X_valid_raw)
        X_test_raw_list.append(X_test_raw)
        X_test_raw_no_valid_list.append(X_test_raw_no_valid)

        y_train_raw_list.append(y_train_raw)
        y_train_raw_no_valid_list.append(y_train_raw_no_valid)
        y_valid_raw_list.append(y_valid_raw)
        y_test_raw_list.append(y_test_raw)
        y_test_raw_no_valid_list.append(y_test_raw_no_valid)
        
        
    categorical_feature_indices_preprocessed = []
    for index, column in enumerate(X_train_list[0].columns):
        if column in numerical_features:
            pass
        else:
            categorical_feature_indices_preprocessed.append(index)

            
    #print('X_train_list[0].columns', X_train_list[0].columns)
    #print('X_train_raw_list[0].columns', X_train_raw_list[0].columns)
    #print('binary_features', binary_features)
    #print('categorical_features_low_cardinality', categorical_features_low_cardinality)
    #print('categorical_features_high_cardinality', categorical_features_high_cardinality)
    #print('numerical_features', numerical_features)
            
            
    return (((X_train_list, y_train_list), 
             (X_train_no_valid_list, y_train_no_valid_list), 
             (X_valid_list, y_valid_list), 
             (X_test_list, y_test_list), 
             (X_test_no_valid_list, y_test_no_valid_list), 
             normalizer_list_list),
            
            ((X_train_raw_list, y_train_raw_list), 
             (X_train_raw_no_valid_list, y_train_raw_no_valid_list), 
             (X_valid_raw_list, y_valid_raw_list), 
             (X_test_raw_list, y_test_raw_list), 
             (X_test_raw_no_valid_list, y_test_raw_no_valid_list)),
            
            ((X_train_list_cv_list, y_train_list_cv_list), 
             (X_train_no_valid_list_cv_list, y_train_no_valid_list_cv_list), 
             (X_valid_list_cv_list, y_valid_list_cv_list), 
             (X_test_list_cv_list, y_test_list_cv_list), 
             (X_test_no_valid_list_cv_list, y_test_no_valid_list_cv_list), 
             normalizer_list_list_cv_list),
            
            ((X_train_raw_list_cv_list, y_train_raw_list_cv_list), 
             (X_train_raw_no_valid_list_cv_list, y_train_raw_no_valid_list_cv_list), 
             (X_valid_raw_list_cv_list, y_valid_raw_list_cv_list), 
             (X_test_raw_list_cv_list, y_test_raw_list_cv_list), 
             (X_test_raw_no_valid_list_cv_list, y_test_raw_no_valid_list_cv_list)),
            
            categorical_feature_indices_preprocessed,
            categorical_feature_indices)



def get_preprocessed_dataset(identifier, 
                             random_seed=42, 
                             config=None,
                             hpo=False,
                             verbosity=0):
               
    if True:
        categorical_indicator = None
        if identifier == 'BIN:CC18_ilpd':
            dataset = openml.datasets.get_dataset(1480)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')      
                   
        
        elif identifier == 'BIN:CC18_madelon':
            dataset = openml.datasets.get_dataset(1485)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')      
            
        elif identifier == 'BIN:CC18_nomao':
            dataset = openml.datasets.get_dataset(1486)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
            
        elif identifier == 'BIN:CC18_ozone-level-8hr':
            dataset = openml.datasets.get_dataset(1487)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')    
            
        elif identifier == 'BIN:CC18_phoneme':
            dataset = openml.datasets.get_dataset(1489)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  

        elif identifier == 'BIN:CC18_qsar-biodeg':
            dataset = openml.datasets.get_dataset(1494)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
            
        elif identifier == 'BIN:CC18_wdbc':
            dataset = openml.datasets.get_dataset(1510)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                    
        elif identifier == 'BIN:CC18_adult':
            dataset = openml.datasets.get_dataset(1590)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                    
        elif identifier == 'BIN:CC18_Bioresponse':
            dataset = openml.datasets.get_dataset(4134)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                    
        elif identifier == 'BIN:CC18_Amazon_employee_access':
            dataset = openml.datasets.get_dataset(4135)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                    
        elif identifier == 'BIN:CC18_PhishingWebsites':
            dataset = openml.datasets.get_dataset(4534)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                    
        elif identifier == 'BIN:CC18_cylinder-bands':
            dataset = openml.datasets.get_dataset(6332)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                                    
        elif identifier == 'BIN:CC18_dresses-sales':
            dataset = openml.datasets.get_dataset(23381)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                                    
        elif identifier == 'BIN:CC18_numerai28.6':
            dataset = openml.datasets.get_dataset(23517)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                                    
                
        elif identifier == 'BIN:CC18_SpeedDating':
            dataset = openml.datasets.get_dataset(40536)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                                    
        elif identifier == 'BIN:CC18_churn':
            dataset = openml.datasets.get_dataset(40701)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                                    
        elif identifier == 'BIN:CC18_tokyo1':
            dataset = openml.datasets.get_dataset(40705)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                                    
        elif identifier == 'BIN:CC18_wilt':
            dataset = openml.datasets.get_dataset(40983)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  
                                
        elif identifier == 'BIN:CC18_climate-model-simulation-crashes':
            dataset = openml.datasets.get_dataset(40994)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

            data = pd.DataFrame(X, columns=attribute_names)
            data['class'] = y

            X_data = data.drop(['class'], axis = 1)
            y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')  

        else:

            raise SystemExit('Unknown key: ' + str(identifier))  

    if config['computation']['subset_size'] is not None:
        original_size = X_data.shape[0]
        if config['computation']['subset_size'] < original_size:        
            data_with_labels = pd.concat([X_data, y_data], axis=1)
            sampled_data = data_with_labels.sample(n=config['computation']['subset_size'], random_state=random_seed, replace=False)
            X_data, y_data = sampled_data.iloc[:,:-1], sampled_data.iloc[:,-1]
        else:
            data_with_labels = pd.concat([X_data, y_data], axis=1)
            sampled_data = data_with_labels.sample(frac=1, random_state=random_seed, replace=False)
            X_data, y_data = sampled_data.iloc[:,:-1], sampled_data.iloc[:,-1]
    else:
        data_with_labels = pd.concat([X_data, y_data], axis=1)
        sampled_data = data_with_labels.sample(frac=1, random_state=random_seed, replace=False)
        X_data, y_data = sampled_data.iloc[:,:-1], sampled_data.iloc[:,-1]    
    
    if 'BIN:' in identifier or 'MULT:' in identifier:
        X_data.reset_index(drop = True, inplace=True)
        y_data.reset_index(drop = True, inplace=True)

        # Set the minimum number of occurrences for each class
        min_occurrences = 10

        # Get the counts of each class in the dataset
        class_counts = y_data.value_counts()

        # Find the classes that occur less than the minimum number of occurrences
        infrequent_classes = class_counts[class_counts < min_occurrences].index.tolist()

        # Remove the rows that correspond to infrequent classes
        X_data = X_data[~y_data.isin(infrequent_classes)]
        y_data = y_data[~y_data.isin(infrequent_classes)]
    
        y_data = pd.Series(OrdinalEncoder().fit_transform(y_data.values.reshape(-1, 1)).flatten(), name=y_data.name)   
        
    X_data.reset_index(drop = True, inplace=True)
    y_data.reset_index(drop = True, inplace=True)

    medians = X_data.median()
    
    X_data = X_data.fillna(medians)

    
    (((X_train_list, y_train_list), 
     (X_train_no_valid_list, y_train_no_valid_list), 
     (X_valid_list, y_valid_list), 
     (X_test_list, y_test_list), 
     (X_test_no_valid_list, y_test_no_valid_list), 
     normalizer_list_list),

    ((X_train_raw_list, y_train_raw_list), 
     (X_train_raw_no_valid_list, y_train_raw_no_valid_list), 
     (X_valid_raw_list, y_valid_raw_list), 
     (X_test_raw_list, y_test_raw_list), 
     (X_test_raw_no_valid_list, y_test_raw_no_valid_list)),

    ((X_train_list_cv, y_train_list_cv), 
     (X_train_no_valid_list_cv, y_train_no_valid_list_cv), 
     (X_valid_list_cv, y_valid_list_cv), 
     (X_test_list_cv, y_test_list_cv), 
     (X_test_no_valid_list_cv, y_test_no_valid_list_cv), 
     normalizer_list_list_cv),

    ((X_train_raw_list_cv, y_train_raw_list_cv), 
     (X_train_raw_no_valid_list_cv, y_train_raw_no_valid_list_cv), 
     (X_valid_raw_list_cv, y_valid_raw_list_cv), 
     (X_test_raw_list_cv, y_test_raw_list_cv), 
     (X_test_raw_no_valid_list_cv, y_test_raw_no_valid_list_cv)),

    categorical_feature_indices_preprocessed,
    categorical_feature_indices) = preprocess_data(X_data, 
                                                   y_data,
                                                   categorical_indicator,
                                                   config,
                                                   random_seed=random_seed,
                                                   verbosity=verbosity,
                                                   hpo=hpo)      

    
    dict_list = []
    for i in range(config['computation']['cv_num_eval']):
        if hpo and config['computation']['cv_num_hpo'] >= 1:
            some_dict = []
            for j in range(config['computation']['cv_num_hpo']):
                some_dict_2 = {
                   'X_train': X_train_list[i],
                   'y_train': y_train_list[i],
                   'X_train_no_valid': X_train_no_valid_list[i],
                   'y_train_no_valid': y_train_no_valid_list[i],                    
                   'X_valid': X_valid_list[i],
                   'y_valid': y_valid_list[i],
                   'X_test': X_test_list[i],
                   'y_test': y_test_list[i],
                   'X_test_no_valid': X_test_no_valid_list[i],
                   'y_test_no_valid': y_test_no_valid_list[i],                    
                   'normalizer_list': None,#normalizer_list_list[i],
                    
                   'X_train_raw': X_train_raw_list[i],
                   'y_train_raw': y_train_raw_list[i],
                   'X_train_raw_no_valid': X_train_raw_no_valid_list[i],
                   'y_train_raw_no_valid': y_train_raw_no_valid_list[i],                    
                   'X_valid_raw': X_valid_raw_list[i],
                   'y_valid_raw': y_valid_raw_list[i],
                   'X_test_raw': X_test_raw_list[i],
                   'y_test_raw': y_test_raw_list[i],
                   'X_test_raw_no_valid': X_test_raw_no_valid_list[i],
                   'y_test_raw_no_valid': y_test_raw_no_valid_list[i],    
                    
                   'X_train_cv': X_train_list_cv[i][j],
                   'y_train_cv': y_train_list_cv[i][j],
                   'X_train_no_valid_cv': X_train_no_valid_list_cv[i][j],
                   'y_train_no_valid_cv': y_train_no_valid_list_cv[i][j], 
                   'X_valid_cv': X_valid_list_cv[i][j],
                   'y_valid_cv': y_valid_list_cv[i][j],
                   'X_test_cv': X_test_list_cv[i][j],
                   'y_test_cv': y_test_list_cv[i][j],
                   'X_test_no_valid_cv': X_test_no_valid_list_cv[i][j],
                   'y_test_no_valid_cv': y_test_no_valid_list_cv[i][j],                    
                   'normalizer_list_cv': None,#normalizer_list_list_cv[i][j]     
                    
                   'X_train_raw_cv': X_train_raw_list_cv[i][j],
                   'y_train_raw_cv': y_train_raw_list_cv[i][j],
                   'X_train_raw_no_valid_cv': X_train_raw_no_valid_list_cv[i][j],
                   'y_train_raw_no_valid_cv': y_train_raw_no_valid_list_cv[i][j], 
                   'X_valid_raw_cv': X_valid_raw_list_cv[i][j],
                   'y_valid_raw_cv': y_valid_raw_list_cv[i][j],
                   'X_test_raw_cv': X_test_raw_list_cv[i][j],
                   'y_test_raw_cv': y_test_raw_list_cv[i][j],
                   'X_test_raw_no_valid_cv': X_test_raw_no_valid_list_cv[i][j],
                   'y_test_raw_no_valid_cv': y_test_raw_no_valid_list_cv[i][j],   
                    
                   'categorical_feature_indices_preprocessed': categorical_feature_indices_preprocessed,
                   'categorical_feature_indices': categorical_feature_indices,
                }
                some_dict.append(some_dict_2)
        else:
            some_dict = {
               'X_train': X_train_list[i],
               'y_train': y_train_list[i],
               'X_train_no_valid': X_train_no_valid_list[i],
               'y_train_no_valid': y_train_no_valid_list[i],                
               'X_valid': X_valid_list[i],
               'y_valid': y_valid_list[i],
               'X_test': X_test_list[i],
               'y_test': y_test_list[i],
               'X_test_no_valid': X_test_no_valid_list[i],
               'y_test_no_valid': y_test_no_valid_list[i],                
               'normalizer_list': None,#normalizer_list_list[i],  
                
               'X_train_raw': X_train_raw_list[i],
               'y_train_raw': y_train_raw_list[i],
               'X_train_raw_no_valid': X_train_raw_no_valid_list[i],
               'y_train_raw_no_valid': y_train_raw_no_valid_list[i],                
               'X_valid_raw': X_valid_raw_list[i],
               'y_valid_raw': y_valid_raw_list[i],
               'X_test_raw': X_test_raw_list[i],
               'y_test_raw': y_test_raw_list[i],
               'X_test_raw_no_valid': X_test_raw_no_valid_list[i],
               'y_test_raw_no_valid': y_test_raw_no_valid_list[i],  
                
               'categorical_feature_indices_preprocessed': categorical_feature_indices_preprocessed,
               'categorical_feature_indices': categorical_feature_indices,                
               }
        dict_list.append(some_dict)

    return dict_list



def evaluate_GRANDE_single(identifier,
                        timestr,
                        dataset_dict,
                          random_seed_data, 
                          random_seed_model, 
                          config_training,
                          benchmark_dict,
                          metrics,
                          hpo=False,
                          verbosity=-1):     
              
    sys.path.append('./utilities/TabSurvey/')

    model_dict = {}
    runtime_dict = {}

    scores_dict = {'GRANDE': {}}  

    number_of_classes = len(np.unique(np.concatenate([dataset_dict['y_train'].values, dataset_dict['y_valid'].values, dataset_dict['y_test'].values]))) if config_training['GRANDE']['objective'] == 'classification' else 1

    for key, value in benchmark_dict.items():
        scores_dict[key] = {}
        try: 
            with timeout(60*60*config_training['computation']['max_hours'], exception=RuntimeError):

                if key == 'CART':
                    if config_training['computation']['use_best_hpo_result']:

                        if config_training['GRANDE']['objective'] == 'classification':
                            model_identifier = 'CART_class'
                        else:
                            model_identifier = 'CART_reg'

                        try:
                            hpo_results = read_best_hpo_result_from_csv_benchmark(identifier, 
                                                                                  model_identifier=model_identifier, 
                                                                                  config=config_training, 
                                                                                  return_best_only=True, 
                                                                                  ascending=False)
                            if verbosity > 1:
                                print('Loaded CART Parameters for ' + identifier + ' with Score ' + str(hpo_results['score']))

                            params = hpo_results['model']

                            if config_training['computation']['force_depth']:
                                params['max_depth'] = config_training['GRANDE']['depth']

                        except FileNotFoundError:
                            print('No Best Parameters CART for ' + identifier)     
                            params = {'max_depth': config_training['GRANDE']['depth'],
                                      'random_state': random_seed_model}                        

                    else:
                        params = {'max_depth': config_training['GRANDE']['depth'],
                                  'random_state': random_seed_model}
                    #print('CART', params)
                    start = timeit.default_timer()

                    if config_training['GRANDE']['objective'] == 'classification':
                         CART_model = DecisionTreeClassifier
                    else:
                         CART_model = DecisionTreeRegressor                   

                    model = CART_model()

                    if 'max_leaf_nodes' in params:
                        #params['max_leaf_nodes'] = int(params['max_leaf_nodes']) if int(params['max_leaf_nodes']) >= 2 else None if isinstance(params['max_leaf_nodes'], numbers.Number) else None
                        params['max_leaf_nodes'] = None if not isinstance(params['max_leaf_nodes'], numbers.Number) else int(params['max_leaf_nodes']) if int(params['max_leaf_nodes']) >= 2 else None

                    if 'min_impurity_decrease' in params:
                        params['min_impurity_decrease'] = 0.0 if params['min_impurity_decrease'] is None else params['min_impurity_decrease']

                    model.set_params(**params)

                    if config_training['GRANDE']['class_weights']:
                        model.fit(enforce_numpy(dataset_dict['X_train_no_valid']), 
                                  enforce_numpy(dataset_dict['y_train_no_valid']),
                                  sample_weight=calculate_sample_weights(enforce_numpy(dataset_dict['y_train_no_valid'])))     
                    else:
                        model.fit(enforce_numpy(dataset_dict['X_train_no_valid']), 
                                  enforce_numpy(dataset_dict['y_train_no_valid']))                                 

                    end = timeit.default_timer()  
                    runtime = end - start
                elif key == 'NeuralNetwork':
                    start = timeit.default_timer()                

                    model = Sequential()
                    model.add(Dense(50, input_shape=(dataset_dict['X_train'].shape[1],), activation='relu'))
                    model.add(Dense(50, activation='relu'))


                    if config_training['GRANDE']['objective'] == 'classification':
                        if number_of_classes == 2:
                            model.add(Dense(1, activation='sigmoid'))
                            model.compile(loss='binary_crossentropy', optimizer='adam')
                        else:
                            model.add(Dense(number_of_classes, activation='softmax'))
                            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
                    else:
                        model.add(Dense(1, activation='linear'))

                        model.compile(loss='mse', optimizer='adam')

                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)

                    history = model.fit(dataset_dict['X_train'], 
                                        dataset_dict['y_train'],
                                        validation_data=(dataset_dict['X_valid'], dataset_dict['y_valid']), 
                                        epochs=1000, 
                                        callbacks=[es], 
                                        verbose=0)        

                    #model.predict = functools.partial(model.predict, verbose = 0)

                    end = timeit.default_timer()  
                    runtime = end - start     

                elif key == 'XGB':
                    start = timeit.default_timer()         

                    if config_training['computation']['use_best_hpo_result']:                             

                        if config_training['GRANDE']['objective'] == 'classification':
                            model_identifier = 'XGB_class'
                        else:
                            model_identifier = 'XGB_reg'    

                        try:
                            hpo_results = read_best_hpo_result_from_csv_benchmark(identifier, 
                                                                                  model_identifier=model_identifier, 
                                                                                  config=config_training, 
                                                                                  return_best_only=True, 
                                                                                  ascending=False)
                            if verbosity > 1:
                                print('Loaded XGB Parameters for ' + identifier + ' with Score ' + str(hpo_results['score']))

                            params = hpo_results['model']

                            if config_training['computation']['force_depth']:
                                params['max_depth'] = config_training['GRANDE']['depth']

                        except FileNotFoundError:
                            print('No Best Parameters XGB for ' + identifier)          
                            params = {
                                     #'learning_rate': 0.1,
                                     'n_estimators': 1_000,#config_training['GRANDE']['n_estimators'],
                                     'early_stopping_rounds': 20,
                                     #'max_depth': config_training['GRANDE']['depth'],
                                     #'min_child_weight': 1,
                                     #'gamma': 0,
                                     #'subsample': 1,                    
                                     #'colsample_bytree': 1,
                                
                                     'max_cat_threshold': 8,
                                     'random_state': random_seed_model,
                                     'n_jobs': 1,
                                     'verbosity': 0,
                                     }                        

                    else:

                        params = {
                                     #'learning_rate': 0.1,
                                     'n_estimators': 1_000,
                                     'early_stopping_rounds': 20,                                        
                                     #'max_depth': config_training['GRANDE']['depth'],
                                     #'min_child_weight': 1,
                                     #'gamma': 0,
                                     #'subsample': 1,                    
                                     #'colsample_bytree': 1,
                            
                                     'max_cat_threshold': 8,
                                     'random_state': random_seed_model,
                                     'n_jobs': 1,   
                                     'verbosity': 0,
                        }

                                
                    if config_training['GRANDE']['objective'] == 'classification':
                         xgb_model = XGBClassifier
                    else:
                         xgb_model = XGBRegressor     

                    if config_training['preprocessing']['XGBoostEncoding']:
                        feature_types = []
                        for feature_index in range(dataset_dict['X_train_raw'].shape[1]):
                            if feature_index in dataset_dict['categorical_feature_indices']:
                                feature_types.append('c')
                            else:
                                feature_types.append('q')

                        model = xgb_model(enable_categorical=True, feature_types=feature_types) 
                    else:
                        model = xgb_model() 
                    
                    params.update({'n_jobs': int(params['n_jobs'])})
                    try:
                        params.update({'max_depth': int(params['max_depth'])})
                    except KeyError:
                        pass
                    params.update({'n_estimators': int(params['n_estimators'])})
                    params.update({'early_stopping_rounds': int(params['early_stopping_rounds'])})     
                    params.update({'verbosity': 0})
                    params.update({'max_cat_threshold': int(params['max_cat_threshold'])})
                    if 'max_cat_to_onehot' in params.keys():
                        params.update({'max_cat_to_onehot': int(params['max_cat_to_onehot'])})
                    
                    try:
                        params.update({'random_state': int(params['random_state'])})
                    except:
                        params.update({'seed': int(params['seed'])})
                                
                    if config_training['computation']['use_gpu']:
                        params.update({'tree_method': 'gpu_hist'})
                    else:
                        params.update({'tree_method': 'hist'})

                    model.set_params(**params)

                    if config_training['preprocessing']['XGBoostEncoding']:
                        if config_training['GRANDE']['class_weights']:
                            model.fit(dataset_dict['X_train_raw'], 
                                      dataset_dict['y_train_raw'],
                                      sample_weight=calculate_sample_weights(enforce_numpy(dataset_dict['y_train_raw'])),
                                      eval_set=[(dataset_dict['X_valid_raw'], dataset_dict['y_valid_raw'])],
                                      sample_weight_eval_set=[calculate_sample_weights(enforce_numpy(dataset_dict['y_valid_raw']))], 
                                      #eval_metric='auc' if config_training['GRANDE']['objective'] == 'classification' else 'mae',
                                      verbose=False)    
                        else:
                            model.fit(dataset_dict['X_train_raw'], 
                                      dataset_dict['y_train_raw'],
                                      eval_set=[(dataset_dict['X_valid_raw'], dataset_dict['y_valid_raw'])],
                                      #eval_metric='auc' if config_training['GRANDE']['objective'] == 'classification' else 'mae',
                                      verbose=False)    
                    else:
                        if config_training['GRANDE']['class_weights']:
                            model.fit(dataset_dict['X_train'], 
                                      dataset_dict['y_train'],
                                      sample_weight=calculate_sample_weights(enforce_numpy(dataset_dict['y_train'])),
                                      eval_set=[(dataset_dict['X_valid'], dataset_dict['y_valid'])],
                                      sample_weight_eval_set=[calculate_sample_weights(enforce_numpy(dataset_dict['y_valid']))], 
                                      #eval_metric='auc' if config_training['GRANDE']['objective'] == 'classification' else 'mae',
                                      verbose=False)    
                        else:
                            model.fit(dataset_dict['X_train'], 
                                      dataset_dict['y_train'],
                                      eval_set=[(dataset_dict['X_valid'], dataset_dict['y_valid'])],
                                      #eval_metric='auc' if config_training['GRANDE']['objective'] == 'classification' else 'mae',
                                      verbose=False)    


                    end = timeit.default_timer()  
                    runtime = end - start     

                elif key == 'CatBoost':
                    start = timeit.default_timer()         

                    if config_training['computation']['use_best_hpo_result']:                             

                        if config_training['GRANDE']['objective'] == 'classification':
                            model_identifier = 'CatBoost_class'
                        else:
                            model_identifier = 'CatBoost_reg'                               

                        try:
                            hpo_results = read_best_hpo_result_from_csv_benchmark(identifier, 
                                                                                  model_identifier=model_identifier, 
                                                                                  config=config_training, 
                                                                                  return_best_only=True, 
                                                                                  ascending=False)

                            if verbosity > 1:
                                print('Loaded CatBoost Parameters for ' + identifier + ' with Score ' + str(hpo_results['score']))

                            params = hpo_results['model']

                            if config_training['computation']['force_depth']:
                                params['max_depth'] = config_training['GRANDE']['depth']

                        except FileNotFoundError:
                            print('No Best Parameters CatBoost for ' + identifier)          
                            params = {
                                     'n_estimators': 1_000,
                                     'early_stopping_rounds': 20,
                                     'random_seed': random_seed_model,
                                     'verbose': 0,      
                                     'boosting_type': 'Plain',
                                     #'one_hot_max_size': 256, 
                                     #'leaf_estimation_iterations': 1,     
                                     #'border_count': 32,
                                
                                     }              

                    else:
                        params = {
                                 'n_estimators': 1_000,
                                 'early_stopping_rounds': 20,
                                 'random_seed': random_seed_model,
                                 'verbose': 0,    
                                 'boosting_type': 'Plain',
                                 #'one_hot_max_size': 256, 
                                 #'leaf_estimation_iterations': 1,  
                                 #'border_count': 32,
                            
                                 }                        

                    if config_training['preprocessing']['CatBoostEncoding']:
                        if config_training['GRANDE']['objective'] == 'classification':
                            if config_training['GRANDE']['class_weights']:
                                catboost_model = CatBoostClassifier
                                train_data = Pool(
                                        data=enforce_numpy(dataset_dict['X_train_raw']),
                                        label=enforce_numpy(dataset_dict['y_train_raw']),
                                        cat_features=dataset_dict['categorical_feature_indices'],
                                        weight=calculate_sample_weights(enforce_numpy(dataset_dict['y_train_raw']))
                                    )

                                eval_data = Pool(
                                        data=enforce_numpy(dataset_dict['X_valid_raw']),
                                        label=enforce_numpy(dataset_dict['y_valid_raw']),
                                        cat_features=dataset_dict['categorical_feature_indices'],
                                        weight=calculate_sample_weights(enforce_numpy(dataset_dict['y_valid_raw']))
                                    )      
                            else:
                                catboost_model = CatBoostClassifier
                                train_data = Pool(
                                        data=enforce_numpy(dataset_dict['X_train_raw']),
                                        label=enforce_numpy(dataset_dict['y_train_raw']),
                                        cat_features=dataset_dict['categorical_feature_indices'],
                                    )

                                eval_data = Pool(
                                        data=enforce_numpy(dataset_dict['X_valid_raw']),
                                        label=enforce_numpy(dataset_dict['y_valid_raw']),
                                        cat_features=dataset_dict['categorical_feature_indices'],
                                    )                                              

                        else:
                            catboost_model = CatBoostRegressor 
                            train_data = Pool(
                                    data=enforce_numpy(dataset_dict['X_train_raw']),
                                    label=enforce_numpy(dataset_dict['y_train_raw']),
                                    cat_features=dataset_dict['categorical_feature_indices'],
                                )

                            eval_data = Pool(
                                    data=enforce_numpy(dataset_dict['X_valid_raw']),
                                    label=enforce_numpy(dataset_dict['y_valid_raw']),
                                    cat_features=dataset_dict['categorical_feature_indices'],
                                )                                       
                    else:
                        if config_training['GRANDE']['objective'] == 'classification':
                            if config_training['GRANDE']['class_weights']:
                                catboost_model = CatBoostClassifier
                                train_data = Pool(
                                        data=enforce_numpy(dataset_dict['X_train']),
                                        label=enforce_numpy(dataset_dict['y_train']),
                                        cat_features=None,
                                        weight=calculate_sample_weights(enforce_numpy(dataset_dict['y_train']))
                                    )

                                eval_data = Pool(
                                        data=enforce_numpy(dataset_dict['X_valid']),
                                        label=enforce_numpy(dataset_dict['y_valid']),
                                        cat_features=None,
                                        weight=calculate_sample_weights(enforce_numpy(dataset_dict['y_valid']))
                                    )      
                            else:
                                catboost_model = CatBoostClassifier
                                train_data = Pool(
                                        data=enforce_numpy(dataset_dict['X_train']),
                                        label=enforce_numpy(dataset_dict['y_train']),
                                        cat_features=None,
                                    )

                                eval_data = Pool(
                                        data=enforce_numpy(dataset_dict['X_valid']),
                                        label=enforce_numpy(dataset_dict['y_valid']),
                                        cat_features=None,
                                    )                                              

                        else:
                            catboost_model = CatBoostRegressor 
                            train_data = Pool(
                                    data=enforce_numpy(dataset_dict['X_train']),
                                    label=enforce_numpy(dataset_dict['y_train']),
                                    cat_features=None,
                                )

                            eval_data = Pool(
                                    data=enforce_numpy(dataset_dict['X_valid']),
                                    label=enforce_numpy(dataset_dict['y_valid']),
                                    cat_features=None,
                                )                                          



                    model = catboost_model()

                    if config_training['computation']['use_gpu']:
                        params.update({'task_type': 'GPU'})
                    else:
                        params.update({'task_type': 'CPU'})

                    params.update({'gpu_ram_part': 0.8/config_training['computation']['n_jobs']/config_training['computation']['number_of_gpus']})
                    params.update({'used_ram_limit': '16GB'})
                    params.update({'thread_count': 1})                                
                    params.update({'devices': '0'}) 
                    params.update({'verbose': int(params['verbose'])})  
                        
                    model.set_params(**params)

                    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                        with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
                            model.fit(X=train_data, eval_set=eval_data)


                    end = timeit.default_timer()  
                    runtime = end - start     

                elif key == 'RandomForest':
                    start = timeit.default_timer()         

                    if config_training['computation']['use_best_hpo_result']:                              

                        if config_training['GRANDE']['objective'] == 'classification':
                            model_identifier = 'RF_class'
                        else:
                            model_identifier = 'RF_reg'                            

                        try:
                            hpo_results = read_best_hpo_result_from_csv_benchmark(identifier, 
                                                                                  model_identifier=model_identifier, 
                                                                                  config=config_training, 
                                                                                  return_best_only=True, 
                                                                                  ascending=False)
                            if verbosity > 1:
                                print('Loaded RandomForest Parameters for ' + identifer + ' with Score ' + str(hpo_results['score']))

                            params = hpo_results['model']

                            if config_training['computation']['force_depth']:
                                params['max_depth'] = config_training['GRANDE']['depth']

                        except FileNotFoundError:
                            print('No Best Parameters RandomForest for ' + identifier)          


                            params = {
                                     #'learning_rate': 0.1,
                                     'n_estimators': config_training['GRANDE']['n_estimators'],
                                     'max_depth': config_training['GRANDE']['depth'],

                                     'criterion': 'gini', #“gini”, “entropy”, “log_loss”
                                     'min_samples_split': 2, 
                                     'min_samples_leaf': 1, 
                                     'min_weight_fraction_leaf': 0.0,

                                     'random_state': random_seed_model,
                                     'n_jobs': 1,
                                     }                        

                    else:
                        params = {
                                     #'learning_rate': 0.1,
                                     'n_estimators': config_training['GRANDE']['n_estimators'],
                                     'max_depth': config_training['GRANDE']['depth'],

                                     'criterion': 'gini', #“gini”, “entropy”, “log_loss”
                                     'min_samples_split': 2, 
                                     'min_samples_leaf': 1, 
                                     'min_weight_fraction_leaf': 0.0,

                                     'random_state': random_seed_model,
                                     'n_jobs': 1,
                                     }                       


                    if 'max_leaf_nodes' in params:
                        #params['max_leaf_nodes'] = int(params['max_leaf_nodes']) if int(params['max_leaf_nodes']) >= 2 else None if isinstance(params['max_leaf_nodes'], numbers.Number) else None
                        params['max_leaf_nodes'] = None if not isinstance(params['max_leaf_nodes'], numbers.Number) else int(params['max_leaf_nodes']) if int(params['max_leaf_nodes']) >= 2 else None

                    if 'min_impurity_decrease' in params:
                        params['min_impurity_decrease'] = 0.0 if params['min_impurity_decrease'] is None else params['min_impurity_decrease']

                    if config_training['GRANDE']['objective'] == 'classification':
                         rf_model = RandomForestClassifier
                    else:
                         rf_model = RandomForestRegressor     

                    model = rf_model()                            
                    model.set_params(**params)

                    if config_training['GRANDE']['class_weights']:
                        model.fit(enforce_numpy(dataset_dict['X_train_no_valid']), 
                                  enforce_numpy(dataset_dict['y_train_no_valid']),
                                  sample_weight=calculate_sample_weights(enforce_numpy(dataset_dict['y_train_no_valid'])))    
                    else:
                        model.fit(enforce_numpy(dataset_dict['X_train_no_valid']), 
                                  enforce_numpy(dataset_dict['y_train_no_valid']))                                

                    end = timeit.default_timer()  
                    runtime = end - start     

                elif key == 'NODE':
                    start = timeit.default_timer()         

                    if config_training['computation']['use_best_hpo_result']:                             


                        try:
                            hpo_results = read_best_hpo_result_from_csv_benchmark(identifier, 
                                                                                  model_identifier=key,
                                                                                  config=config_training, 
                                                                                  return_best_only=True, 
                                                                                  ascending=False)
                            if verbosity > 1:
                                print('Loaded NODE Parameters for ' + identifier + ' with Score ' + str(hpo_results['score']))

                            params = hpo_results['model']

                            if config_training['computation']['force_depth']:
                                params['max_depth'] = config_training['GRANDE']['depth']

                        except FileNotFoundError:
                            print('No Best Parameters NODE for ' + identifier)          
                            params = {
                                      'num_layers': 1,#config_training['GRANDE']['num_layers'],
                                      'total_tree_count': 2048,#np.sum(config_training['GRANDE']['n_estimators']) if isinstance(config_training['GRANDE']['n_estimators'], list) else config_training['GRANDE']['n_estimators']*config_training['GRANDE']['num_layers'],
                                      'tree_depth': 6,#config_training['GRANDE']['depth'],
                                      'tree_output_dim': 2,
                                     }       

                    else:
                        params = {
                                  'num_layers': 1,#config_training['GRANDE']['num_layers'],
                                  'total_tree_count': 2048,#np.sum(config_training['GRANDE']['n_estimators']) if isinstance(config_training['GRANDE']['n_estimators'], list) else config_training['GRANDE']['n_estimators']*config_training['GRANDE']['num_layers'],
                                  'tree_depth': 6,#config_training['GRANDE']['depth'],
                                  'tree_output_dim': 2,
                                 }              


                    number_of_batches = 0
                    batch_size = config_training['GRANDE']['batch_size'] 
                    while True:
                        #number_of_batches = dataset_dict['X_train'].shape[0] // batch_size
                        number_of_batches = np.ceil(dataset_dict['X_train'].shape[0] / batch_size)
                        if number_of_batches < 2:
                            batch_size = batch_size / 2
                        else:
                            break
                    adjusted_batch_size = int(np.floor(dataset_dict['X_train'].shape[0] / number_of_batches))
                    batch_size = adjusted_batch_size
                    
                    batch_size_val = batch_size
                    while True:
                        if dataset_dict['X_valid'].shape[0] // batch_size_val < 1:
                            batch_size_val = dataset_dict['X_valid'].shape[0]
                        else:
                            break
                    
                    args = {'num_features': dataset_dict['X_train'].shape[1],
                                           'objective': 'binary' if config_training['GRANDE']['objective'] == 'classification' and number_of_classes==2 else config_training['GRANDE']['objective'],#config_training['GRANDE']['objective'],
                                           'num_classes': 1 if config_training['GRANDE']['objective'] == 'classification' and number_of_classes==2 else number_of_classes,
                                           'use_gpu': config_training['computation']['use_gpu'],
                                           'data_parallel': False,
                                           'early_stopping_rounds': 20,
                                           'logging_period': number_of_batches,
                                           'batch_size': batch_size,
                                           'val_batch_size': batch_size,
                                           'epochs': 1000}     
                    
                    args = MyDictObject(**args)

                    torch.manual_seed(random_seed_model)
                    random.seed(random_seed_model)
                    np.random.seed(random_seed_model)    
                    model = NODE(params, args)

                    model.fit(X=enforce_numpy(dataset_dict['X_train'], dtype='float'),
                                          y=enforce_numpy(dataset_dict['y_train'], dtype='float'),
                                          X_val=enforce_numpy(dataset_dict['X_valid'], dtype='float'),
                                          y_val=enforce_numpy(dataset_dict['y_valid'], dtype='float')) 

                    end = timeit.default_timer()  
                    runtime = end - start     
        
        except RuntimeError as e:
            model = None
            runtime = '>' + str(config_training['computation']['max_hours'])
        except (MemoryError, tf.errors.ResourceExhaustedError) as e:
            model = None
            runtime = 'OOM' + str(config_training['computation']['max_hours'])                    

        runtime_dict[key] = runtime
        model_dict[key] = model                   
       
    ##############################################################

    if config_training['computation']['use_best_hpo_result']:
        try:
            hpo_results = read_best_hpo_result_from_csv_benchmark(identifier, 
                                                                  model_identifier='GRANDE', 
                                                                  config=config_training, 
                                                                  return_best_only=True, 
                                                                  ascending=False)      

            best_params = hpo_results['model']

            try:
                best_params['dense_layer_identifier'] = string_to_list(best_params['dense_layer_identifier'])
            except:
                pass

            try:
                best_params['num_layers'] = int(best_params['num_layers'])
            except:
                pass

            try:
                best_params['depth'] = int(best_params['depth'])
            except:
                pass            
                
            if verbosity > 1:
                print('Loaded Parameters GRANDE for ' + identifier + ' with Score ' + str(best_params['score']))
            #print('best_params', best_params)

            if 'force_split_decision_activation' not in config_training['computation'].keys():
                config_training['computation']['force_split_decision_activation'] = False

            if 'force_weighting' not in config_training['computation'].keys():
                config_training['computation']['force_weighting'] = False

            for model_param_key, model_param_value in best_params.items():
                if model_param_key == 'depth' and config_training['computation']['force_depth']:
                    if verbosity > 0:
                        print('Setting depth to ' + str(config_training['GRANDE'][model_param_key]))
                    config_training['GRANDE'][model_param_key] = min(model_param_value, config_training['GRANDE'][model_param_key])
                elif model_param_key == 'fine_tune':
                    if verbosity > 0:
                        print('Setting fine_tune to ' + str(config_training['GRANDE'][model_param_key]))
                    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]
                
                elif model_param_key == 'dropout' and config_training['computation']['force_dropout']:
                    if verbosity > 0:
                        print('Setting dropout to ' + str(config_training['GRANDE'][model_param_key]))
                    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]  
                elif model_param_key == 'estimator_leaf_weights' and config_training['computation']['force_weighting']:
                    if verbosity > 0:
                        print('Setting estimator_leaf_weights to ' + str(config_training['GRANDE'][model_param_key]))
                    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]  
                elif model_param_key == 'split_decision_activation' and config_training['computation']['force_split_decision_activation']:
                    if verbosity > 0:
                        print('Setting split_decision_activation to ' + str(config_training['GRANDE'][model_param_key]))
                    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]                  
                
                elif model_param_key == 'class_weights' and config_training['computation']['force_class_weights']:
                    if verbosity > 0:
                        print('Setting class_weights to ' + str(config_training['GRANDE'][model_param_key]))
                    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]                                                
                elif model_param_key == 'restarts' and config_training['computation']['force_restart']:
                    if verbosity > 0:
                        print('Setting restarts to ' + str(config_training['GRANDE'][model_param_key]))
                    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]                             
                #elif model_param_key == 'restart_type' and config_training['computation']['force_restart']:
                #    if verbosity > 0:
                #        print('Setting restart_type to ' + str(config_training['GRANDE'][model_param_key]))
                #    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]    
                elif model_param_key == 'steps':
                    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]    
                elif model_param_key == 'early_stopping_eval_nums':
                    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]    
                elif model_param_key == 'early_stopping_type':
                    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]                            
                elif model_param_key == 'batch_size':
                    config_training['GRANDE'][model_param_key] = config_training['GRANDE'][model_param_key]                           
                else:
                    config_training['GRANDE'][model_param_key] = model_param_value          

        except FileNotFoundError:
            print('No Best Parameters GRANDE for ' + identifier)


    if config_training['preprocessing']['normalization_technique'] == 'mean':
        config_training['GRANDE']['activation'] = None
        
    start_GRANDE = timeit.default_timer()

    dataset_string = '_'.join('_'.join(identifier.split(':')[1].split('.')).split(' '))
    os.makedirs(os.path.dirname("./temp_models/" + timestr + "/"), exist_ok=True)
    for restart_number in range(config_training['GRANDE']['restarts']+1):

        tf.keras.backend.clear_session()

    
        best_model = None
        best_score = np.inf if config_training['GRANDE']['restart_type'] == 'loss' else -np.inf



        (train_data, 
         valid_data, 
         batch_size, 
         batch_size_val, 
         class_weight_dict, 
         loss_function, 
         optimizer_function_dict,
         metrics_GRANDE, 
         callbacks) = pepare_GRANDE_for_training(config_training, dataset_dict, number_of_classes, timestr)

        config_training['GRANDE']['number_of_classes'] = number_of_classes
        config_training['GRANDE']['number_of_variables'] = dataset_dict['X_train'].shape[1]
        config_training['GRANDE']['mean'] = np.mean(dataset_dict['y_train'])
        config_training['GRANDE']['std'] = np.std(dataset_dict['y_train'])
        config_training['GRANDE']['random_seed'] = config_training['computation']['random_seed']

        model = GRANDE(**config_training['GRANDE'])  

        #model.set_params(**config_training['GRANDE'])   
        model.weights_optimizer = optimizer_function_dict['weights_optimizer']
        model.index_optimizer = optimizer_function_dict['index_optimizer']
        model.values_optimizer = optimizer_function_dict['values_optimizer']
        model.leaf_optimizer = optimizer_function_dict['leaf_optimizer']            

        model.compile(loss=loss_function, metrics=metrics_GRANDE)

        history = model.fit(train_data,
                              epochs=config_training['GRANDE']['epochs'], 
                              #batch_size=config_training['GRANDE']['batch_size'], 
                              #steps_per_epoch=dataset_dict['X_train'].shape[0]//64,
                              validation_data=valid_data,#(dataset_dict['X_valid'], dataset_dict['y_valid']), 
                              callbacks=callbacks,
                              #validation_data=(dataset_dict['X_valid'], dataset_dict['y_valid']), callbacks=[early_stopping],  
                              class_weight = class_weight_dict,
                              verbose=0) 
        
        if config_training['GRANDE']['fine_tune']:

            steps_per_epoch=dataset_dict['X_train'].shape[0]//batch_size
            
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                              min_delta=0.001,
                                                              patience=5, 
                                                              restore_best_weights=True)
        
            optimizer_function_dict = {
                'weights_optimizer': get_optimizer_by_name('GradientAccumulator', config_training['GRANDE']['learning_rate_weights']/10, warmup_steps=0, steps_per_epoch=steps_per_epoch, cosine_decay_steps=0),
                'index_optimizer': get_optimizer_by_name('GradientAccumulator', config_training['GRANDE']['learning_rate_index']/10, warmup_steps=0, steps_per_epoch=steps_per_epoch, cosine_decay_steps=0),
                'values_optimizer': get_optimizer_by_name('GradientAccumulator', config_training['GRANDE']['learning_rate_values']/10, warmup_steps=0, steps_per_epoch=steps_per_epoch, cosine_decay_steps=0),
                'leaf_optimizer': get_optimizer_by_name('GradientAccumulator', config_training['GRANDE']['learning_rate_leaf']/10, warmup_steps=0, steps_per_epoch=steps_per_epoch, cosine_decay_steps=0),  
                                      }
        
            callbacks = [early_stopping]    

            if False:
        
                model.weights_optimizer = optimizer_function_dict['weights_optimizer']
                model.index_optimizer = optimizer_function_dict['index_optimizer']
                model.values_optimizer = optimizer_function_dict['values_optimizer']
                model.leaf_optimizer = optimizer_function_dict['leaf_optimizer']  
            
                model.leaf_trainable = False
                model.split_index_trainable = False
                model.split_values_trainable = True
                model.weights_trainable = False
            
                model.compile(loss=loss_function, metrics=metrics_GRANDE)
                
                history = model.fit(train_data,
                                                  epochs=50, 
                                                  validation_data=valid_data,#(dataset_dict['X_valid'], dataset_dict['y_valid']), 
                                                  callbacks=callbacks,
                                                  class_weight = class_weight_dict,
                                                  verbose=0
                                               )   
        
            model.weights_optimizer = optimizer_function_dict['weights_optimizer']
            model.index_optimizer = optimizer_function_dict['index_optimizer']
            model.values_optimizer = optimizer_function_dict['values_optimizer']
            model.leaf_optimizer = optimizer_function_dict['leaf_optimizer']  
        
            model.leaf_trainable = True
            model.split_index_trainable = False
            model.split_values_trainable = False
            model.weights_trainable = False    
        
            model.compile(loss=loss_function, metrics=metrics_GRANDE)
            
            history = model.fit(train_data,
                                              epochs=50, 
                                              validation_data=valid_data,#(dataset_dict['X_valid'], dataset_dict['y_valid']), 
                                              callbacks=callbacks,
                                              class_weight = class_weight_dict,
                                              verbose=0
                                           )   

            if False:
                       
                model.weights_optimizer = optimizer_function_dict['weights_optimizer']
                model.index_optimizer = optimizer_function_dict['index_optimizer']
                model.values_optimizer = optimizer_function_dict['values_optimizer']
                model.leaf_optimizer = optimizer_function_dict['leaf_optimizer']  
                
                model.leaf_trainable = False
                model.split_index_trainable = False
                model.split_values_trainable = False
                model.weights_trainable = True  
            
                model.compile(loss=loss_function, metrics=metrics_GRANDE)
                
                history = model.fit(train_data,
                                                  epochs=50, 
                                                  validation_data=valid_data,#(dataset_dict['X_valid'], dataset_dict['y_valid']), 
                                                  callbacks=callbacks,
                                                  class_weight = class_weight_dict,
                                                  verbose=0
                                               )      

    
        best_score_current_model = min(history.history['val_loss']) if config_training['GRANDE']['restart_type'] == 'loss' else max(history.history['val_' + metrics_GRANDE[0].name])              
        model_params = model.get_params()

        if (best_score_current_model < best_score and  config_training['GRANDE']['restart_type'] == 'loss') or (best_score_current_model > best_score and  config_training['GRANDE']['restart_type'] == 'metric'):
            best_score = best_score_current_model
            best_model = model

    model_dict['GRANDE'] = best_model      
        
    end_GRANDE = timeit.default_timer()
    runtime_dict['GRANDE'] = end_GRANDE - start_GRANDE     
    ##############################################################


    for key in model_dict.keys(): 

        if model_dict[key] is not None:
            if key == 'GRANDE' or key == 'NeuralNetwork' or key == 'DNDT':

                if config_training['GRANDE']['objective']== 'classification':
                    if key == 'GRANDE':
                        batch_size_nn = 128
                        while dataset_dict['X_train'].shape[0] % batch_size_nn == 1 or dataset_dict['X_valid'].shape[0] % batch_size_nn == 1 or dataset_dict['X_test'].shape[0] % batch_size_nn == 1:
                            batch_size_nn = batch_size_nn + 1      
                        dataset_dict['y_test_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_test'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_test'], dtype='float'), batch_size=batch_size_nn, verbose=0)
                        dataset_dict['y_valid_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_valid'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_valid'], dtype='float'), batch_size=batch_size_nn, verbose=0)
                        dataset_dict['y_train_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_train'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_train'], dtype='float'), batch_size=batch_size_nn, verbose=0)                            

                    else:#if key == 'NeuralNetwork'
                        dataset_dict['y_test_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
                        dataset_dict['y_valid_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))
                        dataset_dict['y_train_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))

                    if number_of_classes <= 2:
                        dataset_dict['y_test_' + key] = tf.cast(tf.round(dataset_dict['y_test_' + key + '_proba']), tf.int64)
                        dataset_dict['y_valid_' + key] = tf.cast(tf.round(dataset_dict['y_valid_' + key + '_proba']), tf.int64)
                        dataset_dict['y_train_' + key] = tf.cast(tf.round(dataset_dict['y_train_' + key + '_proba']), tf.int64)                     
                    else:
                        dataset_dict['y_test_' + key] = tf.argmax(dataset_dict['y_test_' + key + '_proba'], axis=1)
                        dataset_dict['y_valid_' + key] = tf.argmax(dataset_dict['y_valid_' + key + '_proba'], axis=1)
                        dataset_dict['y_train_' + key] = tf.argmax(dataset_dict['y_train_' + key + '_proba'], axis=1)   
                else:
                    if key == 'GRANDE':
                        batch_size_nn = 128
                        while dataset_dict['X_train'].shape[0] % batch_size_nn == 1 or dataset_dict['X_valid'].shape[0] % batch_size_nn == 1 or dataset_dict['X_test'].shape[0] % batch_size_nn == 1:
                            batch_size_nn = batch_size_nn + 1      
                        dataset_dict['y_test_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_test'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_test'], dtype='float'), batch_size=batch_size_nn, verbose=0)
                        dataset_dict['y_valid_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_valid'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_valid'], dtype='float'), batch_size=batch_size_nn, verbose=0)
                        dataset_dict['y_train_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_train'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_train'], dtype='float'), batch_size=batch_size_nn, verbose=0)                            

                    else:#if key == 'NeuralNetwork'
                        dataset_dict['y_test_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_test'])))
                        dataset_dict['y_valid_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_valid'])))
                        dataset_dict['y_train_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_train'])))

            else:                      
                if key == 'NODE':
                    if config_training['GRANDE']['objective']== 'classification':
                        dataset_dict['y_test_' + key] = np.round(model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_test']))[:,1])
                        dataset_dict['y_valid_' + key] = np.round(model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_valid']))[:,1])
                        dataset_dict['y_train_' + key] = np.round(model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_train']))[:,1])

                        dataset_dict['y_test_' + key + '_proba'] = model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_test']))
                        dataset_dict['y_valid_' + key + '_proba'] = model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_valid']))
                        dataset_dict['y_train_' + key + '_proba'] = model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_train']))
                    else:
                        dataset_dict['y_test_' + key] = model_dict[key].predict(pd.DataFrame(dataset_dict['X_test'])) #* model_dict['NodeGAM'].preprocessor.y_std + model_dict['NodeGAM'].preprocessor.y_mu
                        dataset_dict['y_valid_' + key] = model_dict[key].predict(pd.DataFrame(dataset_dict['X_valid'])) #* model_dict['NodeGAM'].preprocessor.y_std + model_dict['NodeGAM'].preprocessor.y_mu
                        dataset_dict['y_train_' + key] = model_dict[key].predict(pd.DataFrame(dataset_dict['X_train'])) #* model_dict['NodeGAM'].preprocessor.y_std + model_dict['NodeGAM'].preprocessor.y_mu
                elif key == 'CART' or key == 'RandomForest': 
                    dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test_no_valid'], dtype='float'))
                    dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train_no_valid'], dtype='float'))         
                    dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train_no_valid'], dtype='float'))  

                    if config_training['GRANDE']['objective']== 'classification':
                        dataset_dict['y_test_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_test_no_valid']))
                        dataset_dict['y_valid_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_train_no_valid']))         
                        dataset_dict['y_train_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_train_no_valid']))  
                        
                elif key == 'CatBoost' or key == 'XGB':
                    if (key == 'CatBoost' and config_training['preprocessing']['CatBoostEncoding']) or (key == 'XGB' and config_training['preprocessing']['XGBoostEncoding']):

                        dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test_raw']))
                        dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_valid_raw']))         
                        dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train_raw']))  

                        if config_training['GRANDE']['objective']== 'classification':
                            dataset_dict['y_test_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_test_raw']))
                            dataset_dict['y_valid_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_valid_raw']))         
                            dataset_dict['y_train_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_train_raw']))  
                    else:

                        dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
                        dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))         
                        dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))  

                        if config_training['GRANDE']['objective']== 'classification':
                            dataset_dict['y_test_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_test']))
                            dataset_dict['y_valid_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_valid']))         
                            dataset_dict['y_train_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_train']))  
                        
                else:
                    dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
                    dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))         
                    dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))  

                    if config_training['GRANDE']['objective']== 'classification':
                        dataset_dict['y_test_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_test']))
                        dataset_dict['y_valid_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_valid']))         
                        dataset_dict['y_train_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_train']))  

                        
                if number_of_classes <= 2 and config_training['GRANDE']['objective']== 'classification':
                    dataset_dict['y_test_' + key + '_proba'] = enforce_numpy(dataset_dict['y_test_' + key + '_proba'][:,1])
                    dataset_dict['y_valid_' + key + '_proba'] = enforce_numpy(dataset_dict['y_valid_' + key + '_proba'][:,1])
                    dataset_dict['y_train_' + key + '_proba'] = enforce_numpy(dataset_dict['y_train_' + key + '_proba'][:,1])      

            dataset_dict['y_test_' + key] = np.nan_to_num(dataset_dict['y_test_' + key])
            dataset_dict['y_valid_' + key] = np.nan_to_num(dataset_dict['y_valid_' + key])
            dataset_dict['y_train_' + key] = np.nan_to_num(dataset_dict['y_train_' + key])                  

            if config_training['GRANDE']['objective']== 'classification':
                dataset_dict['y_test_' + key + '_proba'] = np.nan_to_num(dataset_dict['y_test_' + key + '_proba'])
                dataset_dict['y_valid_' + key + '_proba'] = np.nan_to_num(dataset_dict['y_valid_' + key + '_proba'])
                dataset_dict['y_train_' + key + '_proba'] = np.nan_to_num(dataset_dict['y_train_' + key + '_proba'])                  


            if key == 'CART' or key == 'RandomForest':
                y_test_data = dataset_dict['y_test_no_valid']
                y_valid_data = dataset_dict['y_train_no_valid']
                y_train_data = dataset_dict['y_train_no_valid']  
            else:# key == 'GRANDE' or key == 'NeuralNetwork' or key == 'DNDT' or key == 'NODE' or key == 'NodeGAM':
                y_test_data = dataset_dict['y_test']
                y_valid_data = dataset_dict['y_valid']
                y_train_data = dataset_dict['y_train']

            for metric in metrics:
                if config_training['GRANDE']['objective']== 'classification':
                    if metric in ['balanced_accuracy', 'accuracy', 'f1']:
                        y_test = enforce_numpy(np.round(dataset_dict['y_test_' + key]))
                        y_valid = enforce_numpy(np.round(dataset_dict['y_valid_' + key])) 
                        y_train = enforce_numpy(np.round(dataset_dict['y_train_' + key]))  
                    else:
                        y_test = enforce_numpy(dataset_dict['y_test_' + key + '_proba'])
                        y_valid = enforce_numpy(dataset_dict['y_valid_' + key + '_proba'])               
                        y_train = enforce_numpy(dataset_dict['y_train_' + key + '_proba'])    
                else:
                    y_test = enforce_numpy(dataset_dict['y_test_' + key])
                    y_valid = enforce_numpy(dataset_dict['y_valid_' + key])               
                    y_train = enforce_numpy(dataset_dict['y_train_' + key])    

                if metric not in ['f1', 'roc_auc', 'ece', 'AMS']:
                    scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test)
                    scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid)
                    scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train)
                else:          
                    if metric == 'f1':
                        scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test, average='macro')
                        scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid, average='macro')
                        scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train, average='macro')
                    elif metric == 'roc_auc':
                        try:
                            if number_of_classes > 2:                            
                                scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_test_data, num_classes=number_of_classes), y_test, multi_class='ovo')
                            else:
                                scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test)
                        except ValueError:
                            scores_dict[key][metric + '_test'] = 0.5  

                        try:
                            if number_of_classes > 2:                            
                                scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data, num_classes=number_of_classes), y_valid, multi_class='ovo')
                            else:
                                scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid)
                        except ValueError:
                            scores_dict[key][metric + '_valid'] = 0.5       

                        try:
                            if number_of_classes > 2:                            
                                scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_train_data, num_classes=number_of_classes), y_train, multi_class='ovo')
                            else:
                                scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train)
                        except ValueError:
                            scores_dict[key][metric + '_train'] = 0.5                         
                    elif metric == 'ece':
                        if number_of_classes == 2:   
                            scores_dict[key][metric + '_test'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_test_data, tf.int64), logits=tf.cast(tf.stack([1-y_test, y_test], axis=1), tf.float32)).numpy()
                            scores_dict[key][metric + '_valid'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_valid_data, tf.int64), logits=tf.cast(tf.stack([1-y_valid, y_valid], axis=1), tf.float32)).numpy()
                            scores_dict[key][metric + '_train'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_train_data, tf.int64), logits=tf.cast(tf.stack([1-y_train, y_train], axis=1), tf.float32)).numpy()                         
                        else:       
                            scores_dict[key][metric + '_test'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_test_data, tf.int64), logits=tf.cast(y_test, tf.float32)).numpy()
                            scores_dict[key][metric + '_valid'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_valid_data, tf.int64), logits=tf.cast(y_valid, tf.float32)).numpy()
                            scores_dict[key][metric + '_train'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_train_data, tf.int64), logits=tf.cast(y_train, tf.float32)).numpy()   
                    elif metric == 'AMS':
                        scores_dict[key][metric + '_test'] = AMS_metric(y_test_data, y_test)  
                        scores_dict[key][metric + '_valid'] = AMS_metric(y_test_data, y_test)  
                        scores_dict[key][metric + '_train'] = AMS_metric(y_test_data, y_test)                              



                if verbosity > 0:
                    print('Test ' + metric + ' ' + key + ' (' + str(0) + ')', scores_dict[key][metric + '_test'])

        else:

            dataset_dict['y_test_' + key] = np.array([np.nan for _ in range(dataset_dict['X_test'].shape[0])])
            dataset_dict['y_valid_' + key] = np.array([np.nan for _ in range(dataset_dict['X_valid'].shape[0])])
            dataset_dict['y_train_' + key] = np.array([np.nan for _ in range(dataset_dict['X_train'].shape[0])])    

            if config_training['GRANDE']['objective']== 'classification':
                dataset_dict['y_test_' + key + '_proba'] = np.array([[np.nan for _ in range(number_of_classes)] for _ in range(dataset_dict['X_test'].shape[0])])
                dataset_dict['y_valid_' + key + '_proba'] = np.array([[np.nan for _ in range(number_of_classes)] for _ in range(dataset_dict['X_valid'].shape[0])])
                dataset_dict['y_train_' + key + '_proba'] = np.array([[np.nan for _ in range(number_of_classes)] for _ in range(dataset_dict['X_train'].shape[0])])

            for metric in metrics:
                scores_dict[key][metric + '_test'] = np.nan
                scores_dict[key][metric + '_valid'] = np.nan
                scores_dict[key][metric + '_train'] = np.nan        

        scores_dict[key]['runtime'] = runtime_dict[key]        
        
    torch.cuda.empty_cache() 
    del dataset_dict, model_dict
    tf.keras.backend.clear_session()
    gc.collect()
    return identifier, {}, {}, scores_dict


def sleep_minutes(minutes):  
    if minutes > 0:
        for _ in tqdm(range(minutes)):
            time.sleep(60)     

def calculate_scores(model_dict, 
                     dataset_dict,
                     config_training,
                     scores_dict = {},
                     metrics = [],
                     verbosity = 1): 

    number_of_classes = len(np.unique(np.concatenate([dataset_dict['y_train'], dataset_dict['y_valid'], dataset_dict['y_test']]))) if config_training['GRANDE']['objective']== 'classification' else 1

    X_train_GRANDE = dataset_dict['X_train']
    X_valid_GRANDE = dataset_dict['X_valid']
    X_test_GRANDE = dataset_dict['X_test']     

        
        
    for key in model_dict.keys(): 
        
        if model_dict[key] is not None:
            if key == 'GRANDE' or key == 'NeuralNetwork' or key == 'DNDT':
                
                if config_training['GRANDE']['objective']== 'classification':
                    if key == 'GRANDE':
                        batch_size_nn = 128
                        while dataset_dict['X_train'].shape[0] % batch_size_nn == 1 or dataset_dict['X_valid'].shape[0] % batch_size_nn == 1 or dataset_dict['X_test'].shape[0] % batch_size_nn == 1:
                            batch_size_nn = batch_size_nn + 1      
                        dataset_dict['y_test_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_test'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_test'], dtype='float'), batch_size=batch_size_nn, verbose=0)
                        dataset_dict['y_valid_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_valid'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_valid'], dtype='float'), batch_size=batch_size_nn, verbose=0)
                        dataset_dict['y_train_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_train'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_train'], dtype='float'), batch_size=batch_size_nn, verbose=0)                        
                    else:#if key == 'NeuralNetwork'
                        dataset_dict['y_test_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
                        dataset_dict['y_valid_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))
                        dataset_dict['y_train_' + key + '_proba'] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))

                    if number_of_classes <= 2:
                        dataset_dict['y_test_' + key] = tf.cast(tf.round(dataset_dict['y_test_' + key + '_proba']), tf.int64)
                        dataset_dict['y_valid_' + key] = tf.cast(tf.round(dataset_dict['y_valid_' + key + '_proba']), tf.int64)
                        dataset_dict['y_train_' + key] = tf.cast(tf.round(dataset_dict['y_train_' + key + '_proba']), tf.int64)                     
                    else:
                        dataset_dict['y_test_' + key] = tf.argmax(dataset_dict['y_test_' + key + '_proba'], axis=1)
                        dataset_dict['y_valid_' + key] = tf.argmax(dataset_dict['y_valid_' + key + '_proba'], axis=1)
                        dataset_dict['y_train_' + key] = tf.argmax(dataset_dict['y_train_' + key + '_proba'], axis=1)   
                else:
                    if key == 'GRANDE':
                        batch_size_nn = 128
                        while dataset_dict['X_train'].shape[0] % batch_size_nn == 1 or dataset_dict['X_valid'].shape[0] % batch_size_nn == 1 or dataset_dict['X_test'].shape[0] % batch_size_nn == 1:
                            batch_size_nn = batch_size_nn + 1      
                        dataset_dict['y_test_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_test'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_test'], dtype='float'), batch_size=batch_size_nn, verbose=0)
                        dataset_dict['y_valid_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_valid'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_valid'], dtype='float'), batch_size=batch_size_nn, verbose=0)
                        dataset_dict['y_train_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_train'], dtype='float'), batch_size=batch_size_nn, verbose=0)) if number_of_classes <= 2 else model_dict[key].predict(enforce_numpy(dataset_dict['X_train'], dtype='float'), batch_size=batch_size_nn, verbose=0)                        
                    else:#if key == 'NeuralNetwork'
                        dataset_dict['y_test_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_test'])))
                        dataset_dict['y_valid_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_valid'])))
                        dataset_dict['y_train_' + key] = tf.squeeze(model_dict[key].predict(enforce_numpy(dataset_dict['X_train'])))
                    
            else:                       
                if key == 'NODE':
                    if config_training['GRANDE']['objective']== 'classification':
                        dataset_dict['y_test_' + key] = np.round(model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_test']))[:,1])
                        dataset_dict['y_valid_' + key] = np.round(model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_valid']))[:,1])
                        dataset_dict['y_train_' + key] = np.round(model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_train']))[:,1])

                        dataset_dict['y_test_' + key + '_proba'] = model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_test']))
                        dataset_dict['y_valid_' + key + '_proba'] = model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_valid']))
                        dataset_dict['y_train_' + key + '_proba'] = model_dict[key].predict_proba(pd.DataFrame(dataset_dict['X_train']))
                    else:
                        dataset_dict['y_test_' + key] = model_dict[key].predict(pd.DataFrame(dataset_dict['X_test'])) #* model_dict['NodeGAM'].preprocessor.y_std + model_dict['NodeGAM'].preprocessor.y_mu
                        dataset_dict['y_valid_' + key] = model_dict[key].predict(pd.DataFrame(dataset_dict['X_valid'])) #* model_dict['NodeGAM'].preprocessor.y_std + model_dict['NodeGAM'].preprocessor.y_mu
                        dataset_dict['y_train_' + key] = model_dict[key].predict(pd.DataFrame(dataset_dict['X_train'])) #* model_dict['NodeGAM'].preprocessor.y_std + model_dict['NodeGAM'].preprocessor.y_mu

                elif key == 'CART' or key == 'RandomForest' : 
                    dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test_no_valid'], dtype='float'))
                    dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train_no_valid'], dtype='float'))         
                    dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train_no_valid'], dtype='float'))  

                    if config_training['GRANDE']['objective']== 'classification':
                        dataset_dict['y_test_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_test_no_valid']))
                        dataset_dict['y_valid_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_train_no_valid']))         
                        dataset_dict['y_train_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_train_no_valid']))  

                elif key == 'CatBoost' or key == 'XGB':
                    if (key == 'CatBoost' and config_training['preprocessing']['CatBoostEncoding']) or (key == 'XGB' and config_training['preprocessing']['XGBoostEncoding']):                    
                        dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test_raw']))
                        dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_valid_raw']))         
                        dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train_raw']))  

                        if config_training['GRANDE']['objective']== 'classification':
                            dataset_dict['y_test_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_test_raw']))
                            dataset_dict['y_valid_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_valid_raw']))         
                            dataset_dict['y_train_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_train_raw']))
                    else:
                        dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
                        dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))         
                        dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))  

                        if config_training['GRANDE']['objective']== 'classification':
                            dataset_dict['y_test_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_test']))
                            dataset_dict['y_valid_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_valid']))         
                            dataset_dict['y_train_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_train']))                                                  
                            
                        
                else:
                    dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
                    dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))         
                    dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))  

                    if config_training['GRANDE']['objective']== 'classification':
                        dataset_dict['y_test_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_test']))
                        dataset_dict['y_valid_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_valid']))         
                        dataset_dict['y_train_' + key + '_proba'] = model_dict[key].predict_proba(enforce_numpy(dataset_dict['X_train']))  

                if number_of_classes <= 2 and config_training['GRANDE']['objective']== 'classification':
                    dataset_dict['y_test_' + key + '_proba'] = enforce_numpy(dataset_dict['y_test_' + key + '_proba'][:,1])
                    dataset_dict['y_valid_' + key + '_proba'] = enforce_numpy(dataset_dict['y_valid_' + key + '_proba'][:,1])
                    dataset_dict['y_train_' + key + '_proba'] = enforce_numpy(dataset_dict['y_train_' + key + '_proba'][:,1])      


            dataset_dict['y_test_' + key] = np.nan_to_num(dataset_dict['y_test_' + key])
            dataset_dict['y_valid_' + key] = np.nan_to_num(dataset_dict['y_valid_' + key])
            dataset_dict['y_train_' + key] = np.nan_to_num(dataset_dict['y_train_' + key])                  

            if config_training['GRANDE']['objective']== 'classification':
                dataset_dict['y_test_' + key + '_proba'] = np.nan_to_num(dataset_dict['y_test_' + key + '_proba'])
                dataset_dict['y_valid_' + key + '_proba'] = np.nan_to_num(dataset_dict['y_valid_' + key + '_proba'])
                dataset_dict['y_train_' + key + '_proba'] = np.nan_to_num(dataset_dict['y_train_' + key + '_proba'])                  

            if key == 'CART' or key == 'RandomForest':
                y_test_data = dataset_dict['y_test_no_valid']
                y_valid_data = dataset_dict['y_train_no_valid']
                y_train_data = dataset_dict['y_train_no_valid']  
            else:# key == 'GRANDE' or key == 'NeuralNetwork' or key == 'DNDT' or key == 'NODE' or key == 'NodeGAM':
                y_test_data = dataset_dict['y_test']
                y_valid_data = dataset_dict['y_valid']
                y_train_data = dataset_dict['y_train']


            for metric in metrics:
                if config_training['GRANDE']['objective']== 'classification':
                    if metric in ['balanced_accuracy', 'accuracy', 'f1']:
                        y_test = enforce_numpy(np.round(dataset_dict['y_test_' + key]))
                        y_valid = enforce_numpy(np.round(dataset_dict['y_valid_' + key])) 
                        y_train = enforce_numpy(np.round(dataset_dict['y_train_' + key]))  
                    else:
                        y_test = enforce_numpy(dataset_dict['y_test_' + key + '_proba'])
                        y_valid = enforce_numpy(dataset_dict['y_valid_' + key + '_proba'])               
                        y_train = enforce_numpy(dataset_dict['y_train_' + key + '_proba'])    
                else:
                    y_test = enforce_numpy(dataset_dict['y_test_' + key])
                    y_valid = enforce_numpy(dataset_dict['y_valid_' + key])               
                    y_train = enforce_numpy(dataset_dict['y_train_' + key])    
                        
                if metric not in ['f1', 'roc_auc', 'ece', 'AMS']:
                    scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test)
                    scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid)
                    scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train)
                else:          
                    if metric == 'f1':
                        scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test, average='macro')
                        scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid, average='macro')
                        scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train, average='macro')
                    elif metric == 'roc_auc':
                        try:
                            if number_of_classes > 2:                            
                                scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_test_data, num_classes=number_of_classes), y_test, multi_class='ovo')
                            else:
                                scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test)
                        except ValueError:
                            scores_dict[key][metric + '_test'] = 0.5  

                        try:
                            if number_of_classes > 2:                            
                                scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data, num_classes=number_of_classes), y_valid, multi_class='ovo')
                            else:
                                scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid)
                        except ValueError:
                            scores_dict[key][metric + '_valid'] = 0.5       

                        try:
                            if number_of_classes > 2:                            
                                scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_train_data, num_classes=number_of_classes), y_train, multi_class='ovo')
                            else:
                                scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train)
                        except ValueError:
                            scores_dict[key][metric + '_train'] = 0.5                         
                    elif metric == 'ece':
                        if number_of_classes == 2:   
                            scores_dict[key][metric + '_test'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_test_data, tf.int64), logits=tf.cast(tf.stack([1-y_test, y_test], axis=1), tf.float32)).numpy()
                            scores_dict[key][metric + '_valid'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_valid_data, tf.int64), logits=tf.cast(tf.stack([1-y_valid, y_valid], axis=1), tf.float32)).numpy()
                            scores_dict[key][metric + '_train'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_train_data, tf.int64), logits=tf.cast(tf.stack([1-y_train, y_train], axis=1), tf.float32)).numpy()                         
                        else:       
                            scores_dict[key][metric + '_test'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_test_data, tf.int64), logits=tf.cast(y_test, tf.float32)).numpy()
                            scores_dict[key][metric + '_valid'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_valid_data, tf.int64), logits=tf.cast(y_valid, tf.float32)).numpy()
                            scores_dict[key][metric + '_train'] = -expected_calibration_error(num_bins = 10, labels_true=tf.cast(y_train_data, tf.int64), logits=tf.cast(y_train, tf.float32)).numpy()   
                    elif metric == 'AMS':
                        scores_dict[key][metric + '_test'] = AMS_metric(y_test_data, y_test)  
                        scores_dict[key][metric + '_valid'] = AMS_metric(y_test_data, y_test)  
                        scores_dict[key][metric + '_train'] = AMS_metric(y_test_data, y_test)                              
                            
                            
                            
                if verbosity > 0:
                    print('Test ' + metric + ' ' + key + ' (' + str(0) + ')', scores_dict[key][metric + '_test'])

        else:

            dataset_dict['y_test_' + key] = np.array([np.nan for _ in range(dataset_dict['X_test'].shape[0])])
            dataset_dict['y_valid_' + key] = np.array([np.nan for _ in range(dataset_dict['X_valid'].shape[0])])
            dataset_dict['y_train_' + key] = np.array([np.nan for _ in range(dataset_dict['X_train'].shape[0])])    

            if config_training['GRANDE']['objective']== 'classification':
                dataset_dict['y_test_' + key + '_proba'] = np.array([[np.nan for _ in range(number_of_classes)] for _ in range(dataset_dict['X_test'].shape[0])])
                dataset_dict['y_valid_' + key + '_proba'] = np.array([[np.nan for _ in range(number_of_classes)] for _ in range(dataset_dict['X_valid'].shape[0])])
                dataset_dict['y_train_' + key + '_proba'] = np.array([[np.nan for _ in range(number_of_classes)] for _ in range(dataset_dict['X_train'].shape[0])])

            for metric in metrics:
                scores_dict[key][metric + '_test'] = np.nan
                scores_dict[key][metric + '_valid'] = np.nan
                scores_dict[key][metric + '_train'] = np.nan        

        if verbosity > 0:
            print('________________________________________________________________________________________________________')   

def evaluate_GRANDE(identifier,
                 timestr,
                 dataset_dict_list,
                  random_seed_data=42, 
                  random_seed_model=42, 
                  config=None,
                  benchmark_dict_list={},
                  metrics=[],
                  hpo=False,
                  verbosity=0,
                  parallel_jobs_cv=2,
                  parallel_jobs_parent=2):
    
    print('START', identifier)
    
    config_training = deepcopy(config)  
    
    if 'REG:' in identifier:
        config_training['GRANDE']['objective'] = 'regression'
        if 'loss' not in config_training['GRANDE']:
            config_training['GRANDE']['loss'] = 'mse'   
        elif 'crossentropy' in config_training['GRANDE']['loss']:
            config_training['GRANDE']['loss'] = 'mse'       
    elif 'BIN:' in identifier:
        config_training['GRANDE']['objective'] = 'classification'
        if 'loss' not in config_training['GRANDE']:
            config_training['GRANDE']['loss'] = 'crossentropy'     
    elif 'MULT:' in identifier:
        config_training['GRANDE']['objective'] = 'classification'    
        if 'loss' not in config_training['GRANDE']:
            config_training['GRANDE']['loss'] = 'crossentropy'         
    if verbosity > 0:
        print('________________________________________________________________________________________________________')   
        
    #dataset_dict = {}
    #model_dict = {}
    #runtime_dict = {}
    #cv_performance_dict = {}

    #scores_dict = {'GRANDE': {}}
                
    num_gpu = 1/(np.ceil(parallel_jobs_cv*parallel_jobs_parent))  
    
    if dataset_dict_list is None:
        dataset_dict_list = get_preprocessed_dataset(identifier,
                                                random_seed=random_seed_data,
                                                config=config_training,
                                                verbosity=verbosity,
                                                hpo=hpo)       
      
    #with joblib.parallel_backend('ray', ray_remote_args={'num_gpus':0.01}):#ray_remote_args=dict(num_gpus=1) #, 'num_cpus':1 #
    #with joblib.parallel_backend('ray'):#ray_remote_args=dict(num_gpus=1) #, 'num_cpus':1 #
    parallel_eval = Parallel(n_jobs = parallel_jobs_cv, verbose=0, backend='loky') #loky #sequential multiprocessing
    evaluation_results = parallel_eval(delayed(evaluate_GRANDE_single)(identifier=identifier,
                                                                    timestr=timestr,
                                                                    dataset_dict=dataset_dict,
                                                                    random_seed_data=random_seed_data, 
                                                                    random_seed_model=random_seed_model, 
                                                                    config_training=config_training,
                                                                    benchmark_dict=benchmark_dict_list[i],
                                                                    metrics=metrics,
                                                                    hpo=hpo,
                                                                    verbosity=-1) for i, dataset_dict in enumerate(dataset_dict_list))
    print('END', identifier)
    
    return evaluation_results
                



class Letor_Converter(object):    
    '''
    Class Converter implements parsing from original letor txt files to
    pandas data frame representation.
    '''
    
    def __init__(self, path):
        
        '''
        Arguments:
            path: path to letor txt file
        '''
        self._path = path
        
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, p):
        self._path = p
        
    def _load_file(self):
        '''        
        Loads and parses raw letor txt file.
        
        Return:
            letor txt file parsed to csv in raw format
        '''
        return pd.read_csv(str(self._path), sep=" ", header=None)
        
    def _drop_col(self, df):
        '''
        Drops last column, which was added in the parsing procedure due to a
        trailing white space for each sample in the text file
        
        Arguments:
            df: pandas dataframe
        Return:
            df: original df with last column dropped
        '''
        return df.drop(df.columns[-1], axis=1)
    
    def _split_colon(self, df):
        '''
        Splits the data on the colon and transforms it into a tabular format
        where columns are features and rows samples. Cells represent feature
        values per sample.
        
        Arguments:
            df: pandas dataframe object
        Return:
            df: original df with string pattern ':' removed; columns named appropriately
        '''
        for col in range(1,len(df.columns)):
            df.loc[:,col] = df.loc[:,col].apply(lambda x: str(x).split(':')[1])
        df.columns = ['rel', 'qid'] + [str(x) for x in range(1,len(df.columns)-1)] # renaming cols
        return df
    
    def convert(self):
        '''
        Performs final conversion.
        
        Return:
            fully converted pandas dataframe
        '''
        df_raw = self._load_file()
        df_drop = self._drop_col(df_raw)
        return self._split_colon(df_drop)
    


# Read idx file format (from LibSVM)   https://github.com/StephanLorenzen/MajorityVoteBounds/blob/278a2811774e48093a7593e068e5958832cfa686/mvb/data.py
def _read_idx_file(path, d, sep=None, identifier=False):
    X = []
    Y = []
    with open(path) as f:
        for l in f:
            x = np.zeros(d)
            l = l.strip().split() if sep is None else l.strip().split(sep)
            Y.append(int(l[0]))
            for pair in l[1:]:
                pair = pair.strip()
                if pair=='':
                    continue
                (i,v) = pair.split(":")
                if v=='':
                    import pdb; pdb.set_trace()
                    
                if identifier == False:
                    x[int(i)-1] = float(v)
                else:
                    try:
                        x[int(i)] = float(v)
                    except:
                        x[0] = float(v)
            X.append(x)
    return np.array(X),np.array(Y)

def enforce_numpy(data, dtype=None):
    
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        if dtype is None:
            data_numpy = data.to_numpy(dtype=data.dtypes)
        else:
            data_numpy = data.values
    elif tf.is_tensor(data):
        data_numpy = data.numpy()
    else:
        data_numpy = data
        
    return data_numpy

def get_columns_by_name(df, columnname):
    columns = list(df.columns)
    columns_slected = [name for name in columns if columnname == ' '.join(name.split(' ')[1:])]
    return df[columns_slected]

def one_hot_encode_data(df, transformer=None):
    
    if transformer is None:
        transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), df.columns)], remainder='passthrough', sparse_threshold=0)
        transformer.fit(df)        

        df_values = transformer.transform(df)
        df = pd.DataFrame(df_values, columns=transformer.get_feature_names())
        
        return df, transformer
    else:
        df_values = transformer.transform(df)
        df = pd.DataFrame(df_values, columns=transformer.get_feature_names())
        
    return df


def structure_hpo_results_complete(model_identifier_list,
                                   hpo_results_train, 
                                   hpo_results_unsorted_train,                                    
                                   hpo_results_valid, 
                                   hpo_results_unsorted_valid, 
                                   hpo_results_test, 
                                   hpo_results_unsorted_test,
                                   hpo_results_cv, 
                                   hpo_results_unsorted_cv,                                   
                                   metrics): #CURRENTLY EXCLUDES DNDT
    
    results_train = []
    columns_train = []
    
    too_many_variables = False
    for model_identifier in model_identifier_list:
        if model_identifier == 'GRANDE':
            results_train.append([hpo_results_train[key][0]['GRANDE mean (mean)'] for key in hpo_results_train.keys()])
        else:
            if model_identifier == 'DNDT':
                try:
                    results_train.append([hpo_results_train[key][0][model_identifier + ' mean'] for key in hpo_results_train.keys()])
                except:
                    too_many_variables = True
                    break
            else:
                results_train.append([hpo_results_train[key][0][model_identifier + ' mean'] for key in hpo_results_train.keys()])
        columns_train.append(model_identifier + ' (mean)')
        
        results_train.append([hpo_results_train[key][0][model_identifier + ' runtime mean'] for key in hpo_results_train.keys()])
        columns_train.append(model_identifier + ' runtime (mean)')

    best_results_train = pd.DataFrame(data=np.vstack(results_train).T, index=list(hpo_results_train.keys()), columns=columns_train)
    best_results_mean_train = best_results_train.mean()
    best_results_mean_train.name = 'MEAN'
    
    count_list = []
    for column in best_results_train.columns:
        column_metric_identifier = ' '.join(column.split(' ')[1:])
        count_series = best_results_train[column]>=get_columns_by_name(best_results_train, column_metric_identifier).max(axis=1)
        #count_series.drop('MEAN', inplace=True)
        count = count_series.sum()
        count_list.append(count)
    best_results_count_train = pd.Series(count_list, index = best_results_train.columns)
    best_results_count_train.name = 'COUNT'   
    
    best_results_train = best_results_train.append(best_results_mean_train)
    best_results_train = best_results_train.append(best_results_count_train)
    
    if not too_many_variables:
        #reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        #reorder = flatten_list([[i*len(metrics)+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(2)])    
        best_results_train = best_results_train.iloc[:,reorder]
    else:
        #reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        #reorder = flatten_list([[i*len(metrics)+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list)-1)] for j in range(2)])    
        best_results_train = best_results_train.iloc[:,reorder]    
    #######################################################################################################################################    
    
    results_valid = []
    columns_valid = []
    for model_identifier in model_identifier_list:
        if model_identifier == 'GRANDE':
            results_valid.append([hpo_results_valid[key][0]['GRANDE mean (mean)'] for key in hpo_results_valid.keys()])
        else:
            if model_identifier == 'DNDT':
                try:
                    results_valid.append([hpo_results_valid[key][0][model_identifier + ' mean'] for key in hpo_results_valid.keys()])
                except:
                    too_many_variables = True
                    break
            else:
                results_valid.append([hpo_results_valid[key][0][model_identifier + ' mean'] for key in hpo_results_valid.keys()])
            
        columns_valid.append(model_identifier + ' (mean)')
        
        results_valid.append([hpo_results_valid[key][0][model_identifier + ' runtime mean'] for key in hpo_results_valid.keys()])
        columns_valid.append(model_identifier + ' runtime (mean)')

    best_results_valid = pd.DataFrame(data=np.vstack(results_valid).T, index=list(hpo_results_valid.keys()), columns=columns_valid)
    best_results_mean_valid = best_results_valid.mean()
    best_results_mean_valid.name = 'MEAN'
    
    count_list = []
    for column in best_results_valid.columns:
        column_metric_identifier = ' '.join(column.split(' ')[1:])
        count_series = best_results_valid[column]>=get_columns_by_name(best_results_valid, column_metric_identifier).max(axis=1)
        #count_series.drop('MEAN', inplace=True)
        count = count_series.sum()
        count_list.append(count)
    best_results_count_valid = pd.Series(count_list, index = best_results_valid.columns)
    best_results_count_valid.name = 'COUNT'   
    
    best_results_valid = best_results_valid.append(best_results_mean_valid)
    best_results_valid = best_results_valid.append(best_results_count_valid)
    
    if not too_many_variables:
        #reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        #reorder = flatten_list([[i*len(metrics)+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(2)])    
        best_results_valid = best_results_valid.iloc[:,reorder]
    else:
        #reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        #reorder = flatten_list([[i*len(metrics)+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list)-1)] for j in range(2)])    
        best_results_valid = best_results_valid.iloc[:,reorder]        
    
    #######################################################################################################################################
        
    results_test = []
    columns_test = []
    for model_identifier in model_identifier_list:
        if model_identifier == 'GRANDE':
            results_test.append([hpo_results_test[key][0]['GRANDE mean (mean)'] for key in hpo_results_test.keys()])
        else:
            if model_identifier == 'DNDT':
                try:
                    results_test.append([hpo_results_test[key][0][model_identifier + ' mean'] for key in hpo_results_test.keys()])
                except:
                    break
            else:
                results_test.append([hpo_results_test[key][0][model_identifier + ' mean'] for key in hpo_results_test.keys()])
            
        columns_test.append(model_identifier + ' (mean)')
        
        results_test.append([hpo_results_test[key][0][model_identifier + ' runtime mean'] for key in hpo_results_test.keys()])
        columns_test.append(model_identifier + ' runtime (mean)')   
        
    best_results_test = pd.DataFrame(data=np.vstack([results_test]).T, index=list(hpo_results_test.keys()), columns=columns_test)
    best_results_mean_test = best_results_test.mean()
    best_results_mean_test.name = 'MEAN'

    count_list = []
    for column in best_results_test.columns:
        column_metric_identifier = ' '.join(column.split(' ')[1:])
        count_series = best_results_test[column]>=get_columns_by_name(best_results_test, column_metric_identifier).max(axis=1)
        #count_series.drop('MEAN', inplace=True)
        count = count_series.sum()
        count_list.append(count)
    best_results_count_test = pd.Series(count_list, index = best_results_test.columns)
    best_results_count_test.name = 'COUNT' 
    
    best_results_test = best_results_test.append(best_results_mean_test)    
    best_results_test = best_results_test.append(best_results_count_test)    

    if not too_many_variables:
        #reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        #reorder = flatten_list([[i*len(metrics)+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(2)])    
        best_results_test = best_results_test.iloc[:,reorder]
    else:
        #reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        #reorder = flatten_list([[i*len(metrics)+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list)-1)] for j in range(2)])    
        best_results_test = best_results_test.iloc[:,reorder]    
    
    #######################################################################################################################################
        
    results_cv = []
    columns_cv = []
    for model_identifier in model_identifier_list:
        if model_identifier == 'GRANDE':
            results_cv.append([hpo_results_cv[key][0]['GRANDE mean (mean)'] for key in hpo_results_cv.keys()])
        else:
            if model_identifier == 'DNDT':
                try:
                    results_cv.append([hpo_results_cv[key][0][model_identifier + ' mean'] for key in hpo_results_cv.keys()])
                except:
                    too_many_variables = True
                    break
            else:
                results_cv.append([hpo_results_cv[key][0][model_identifier + ' mean'] for key in hpo_results_cv.keys()])  
            
        columns_cv.append(model_identifier + ' (mean)')
        
        results_cv.append([hpo_results_cv[key][0][model_identifier + ' runtime mean'] for key in hpo_results_cv.keys()])
        columns_cv.append(model_identifier + ' runtime (mean)')   
        
    best_results_cv = pd.DataFrame(data=np.vstack([results_cv]).T, index=list(hpo_results_cv.keys()), columns=columns_cv)
    best_results_mean_cv = best_results_cv.mean()
    best_results_mean_cv.name = 'MEAN'

    count_list = []
    for column in best_results_cv.columns:
        column_metric_identifier = ' '.join(column.split(' ')[1:])
        count_series = best_results_cv[column]>=get_columns_by_name(best_results_cv, column_metric_identifier).max(axis=1)
        #count_series.drop('MEAN', inplace=True)
        count = count_series.sum()
        count_list.append(count)
    best_results_count_cv = pd.Series(count_list, index = best_results_cv.columns)
    best_results_count_cv.name = 'COUNT' 
    
    best_results_cv = best_results_cv.append(best_results_mean_cv)    
    best_results_cv = best_results_cv.append(best_results_count_cv)    

    if not too_many_variables:
        #reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        #reorder = flatten_list([[i*len(metrics)+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(2)])    
        best_results_cv = best_results_cv.iloc[:,reorder]
    else:
        #reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        #reorder = flatten_list([[i*len(metrics)+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list)-1)] for j in range(2)])    
        best_results_cv = best_results_cv.iloc[:,reorder]   
    
    
    
    return best_results_train, best_results_valid, best_results_test, best_results_cv
    



def structure_hpo_results_for_dataset(evaluation_results_hpo, 
                                      model_identifier_list, 
                                      hpo_results_train, 
                                      hpo_results_unsorted_train,                                       
                                      hpo_results_valid, 
                                      hpo_results_unsorted_valid, 
                                      hpo_results_test, 
                                      hpo_results_unsorted_test,   
                                      hpo_results_cv, 
                                      hpo_results_unsorted_cv,                                        
                                      comparator_metric='f1', 
                                      greater_better=True,
                                      config=None,
                                      identifier=None):

    mean_list_unsorted_valid = {}
    mean_list_valid = {}
    mean_list_unsorted_test = {}
    mean_list_test = {}
    
    mean_list_unsorted_train = {}
    mean_list_train = {}
    mean_list_unsorted_cv = {}
    mean_list_cv = {}
    
    runtime_dict_unsorted = {}
    runtime_dict = {}
    for model_identifer in model_identifier_list:
        
        if model_identifer == 'GRANDE':
            mean_list_unsorted_cv[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['cv'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(len(evaluation_results_hpo))]
            mean_list_cv[model_identifer] = sorted(mean_list_unsorted_cv[model_identifer], reverse=greater_better)
            mean_list_unsorted_cv[model_identifer + '_max'] = [np.mean(evaluation_results_hpo[i][0]['cv'][model_identifer + ' '  + comparator_metric + '_max']) for i in range(len(evaluation_results_hpo))]
            mean_list_cv[model_identifer + '_max'] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv[model_identifer], mean_list_unsorted_cv[model_identifer + '_max']), key=lambda pair: pair[0], reverse=greater_better)]                  
            
            mean_list_unsorted_train[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['train'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(len(evaluation_results_hpo))]
            mean_list_train[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv[model_identifer], mean_list_unsorted_train[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]
            mean_list_unsorted_train[model_identifer + '_max'] = [np.mean(evaluation_results_hpo[i][0]['train'][model_identifer + ' '  + comparator_metric + '_max']) for i in range(len(evaluation_results_hpo))]
            mean_list_train[model_identifer + '_max'] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv[model_identifer], mean_list_unsorted_train[model_identifer + '_max']), key=lambda pair: pair[0], reverse=greater_better)]            
            
            mean_list_unsorted_valid[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['valid'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(len(evaluation_results_hpo))]
            mean_list_valid[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv[model_identifer], mean_list_unsorted_valid[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]
            mean_list_unsorted_valid[model_identifer + '_max'] = [np.mean(evaluation_results_hpo[i][0]['valid'][model_identifer + ' '  + comparator_metric + '_max']) for i in range(len(evaluation_results_hpo))]
            mean_list_valid[model_identifer + '_max'] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv[model_identifer], mean_list_unsorted_valid[model_identifer + '_max']), key=lambda pair: pair[0], reverse=greater_better)]            
            
            mean_list_unsorted_test[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['test'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(len(evaluation_results_hpo))]
            mean_list_test[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv[model_identifer], mean_list_unsorted_test[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]
            mean_list_unsorted_test[model_identifer + '_max'] = [np.mean(evaluation_results_hpo[i][0]['test'][model_identifer + ' '  + comparator_metric + '_max']) for i in range(len(evaluation_results_hpo))]
            mean_list_test[model_identifer + '_max'] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv[model_identifer], mean_list_unsorted_test[model_identifer + '_max']), key=lambda pair: pair[0], reverse=greater_better)]

        else:
            mean_list_unsorted_cv[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['cv'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(len(evaluation_results_hpo))]
            mean_list_cv[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv['GRANDE'], mean_list_unsorted_cv[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]
            
            mean_list_unsorted_train[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['train'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(len(evaluation_results_hpo))]
            mean_list_train[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv['GRANDE'], mean_list_unsorted_train[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]
            
            mean_list_unsorted_valid[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['valid'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(len(evaluation_results_hpo))]
            mean_list_valid[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv['GRANDE'], mean_list_unsorted_valid[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]
            
            mean_list_unsorted_test[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['test'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(len(evaluation_results_hpo))]
            mean_list_test[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_cv['GRANDE'], mean_list_unsorted_test[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]
            
        if isinstance(evaluation_results_hpo[0][0]['valid'][model_identifer + ' mean runtime'], pd.DataFrame):
            runtime_dict_unsorted[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['valid'][model_identifer + ' mean runtime'].iloc[:,1]) for i in range(len(evaluation_results_hpo))]
        else:
            runtime_dict_unsorted[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['valid'][model_identifer + ' mean runtime']) for i in range(len(evaluation_results_hpo))]
        
        runtime_dict[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_valid['GRANDE'], runtime_dict_unsorted[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]  

    parameter_setting_list_unsorted = [evaluation_results_hpo[i][1] for i in range(len(evaluation_results_hpo))]
    parameter_setting_list = [x for (y,x) in sorted(zip(mean_list_unsorted_cv['GRANDE'], parameter_setting_list_unsorted), key=lambda pair: pair[0], reverse=greater_better)]

    parameter_setting_list_complete_unsorted = [evaluation_results_hpo[i][3] for i in range(len(evaluation_results_hpo))]#[list(evaluation_results_hpo[i][2].values())[0]['GRANDE'][0].get_params() for i in range(config['computation']['search_iterations'])]
    parameter_setting_list_complete = [x for (y,x) in sorted(zip(mean_list_unsorted_cv['GRANDE'], parameter_setting_list_complete_unsorted), key=lambda pair: pair[0], reverse=greater_better)]      
    
    
    hpo_results_by_identifer = []
    for i in range(len(evaluation_results_hpo)):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GRANDE':
                result_dict['GRANDE mean (mean)'] = mean_list_train['GRANDE'][i]
                result_dict['GRANDE max (mean)'] = mean_list_train['GRANDE_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_train[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete[i]

        hpo_results_by_identifer.append(result_dict)
    hpo_results_train[identifier] = hpo_results_by_identifer


    hpo_results_by_identifer_unsorted = []
    for i in range(len(evaluation_results_hpo)):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GRANDE':
                result_dict['GRANDE mean (mean)'] = mean_list_unsorted_train['GRANDE'][i]
                result_dict['GRANDE max (mean)'] = mean_list_unsorted_train['GRANDE_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_unsorted_train[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict_unsorted[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list_unsorted[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete_unsorted[i]

        hpo_results_by_identifer_unsorted.append(result_dict)
    hpo_results_unsorted_train[identifier] = hpo_results_by_identifer_unsorted      

    
    
    hpo_results_by_identifer = []
    for i in range(len(evaluation_results_hpo)):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GRANDE':
                result_dict['GRANDE mean (mean)'] = mean_list_valid['GRANDE'][i]
                result_dict['GRANDE max (mean)'] = mean_list_valid['GRANDE_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_valid[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete[i]

        hpo_results_by_identifer.append(result_dict)
    hpo_results_valid[identifier] = hpo_results_by_identifer


    hpo_results_by_identifer_unsorted = []
    for i in range(len(evaluation_results_hpo)):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GRANDE':
                result_dict['GRANDE mean (mean)'] = mean_list_unsorted_valid['GRANDE'][i]
                result_dict['GRANDE max (mean)'] = mean_list_unsorted_valid['GRANDE_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_unsorted_valid[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict_unsorted[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list_unsorted[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete_unsorted[i]

        hpo_results_by_identifer_unsorted.append(result_dict)
    hpo_results_unsorted_valid[identifier] = hpo_results_by_identifer_unsorted      

    ############
    hpo_results_by_identifer = []
    for i in range(len(evaluation_results_hpo)):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GRANDE':
                result_dict['GRANDE mean (mean)'] = mean_list_test['GRANDE'][i]
                result_dict['GRANDE max (mean)'] = mean_list_test['GRANDE_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_test[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete[i]

        hpo_results_by_identifer.append(result_dict)
    hpo_results_test[identifier] = hpo_results_by_identifer


    hpo_results_by_identifer_unsorted = []
    for i in range(len(evaluation_results_hpo)):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GRANDE':
                result_dict['GRANDE mean (mean)'] = mean_list_unsorted_test['GRANDE'][i]
                result_dict['GRANDE max (mean)'] = mean_list_unsorted_test['GRANDE_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_unsorted_test[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict_unsorted[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list_unsorted[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete_unsorted[i]

        hpo_results_by_identifer_unsorted.append(result_dict)
    hpo_results_unsorted_test[identifier] = hpo_results_by_identifer_unsorted

    
    hpo_results_by_identifer = []
    for i in range(len(evaluation_results_hpo)):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GRANDE':
                result_dict['GRANDE mean (mean)'] = mean_list_cv['GRANDE'][i]
                result_dict['GRANDE max (mean)'] = mean_list_cv['GRANDE_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_cv[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete[i]

        hpo_results_by_identifer.append(result_dict)
    hpo_results_cv[identifier] = hpo_results_by_identifer


    hpo_results_by_identifer_unsorted = []
    for i in range(len(evaluation_results_hpo)):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GRANDE':
                result_dict['GRANDE mean (mean)'] = mean_list_unsorted_cv['GRANDE'][i]
                result_dict['GRANDE max (mean)'] = mean_list_unsorted_cv['GRANDE_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_unsorted_cv[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict_unsorted[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list_unsorted[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete_unsorted[i]

        hpo_results_by_identifer_unsorted.append(result_dict)
    hpo_results_unsorted_cv[identifier] = hpo_results_by_identifer_unsorted          
    
    #print('')
    #display(hpo_results_by_identifer[:1])
    #print('___________________________________________________________________________')

    return  hpo_results_train, hpo_results_unsorted_train, hpo_results_valid, hpo_results_unsorted_valid, hpo_results_test, hpo_results_unsorted_test, hpo_results_cv, hpo_results_unsorted_cv
   

def get_params_gentree(tree, config):
    
    return {
                 'n_thresholds': tree._n_thresholds,
                 'n_trees': tree.selector.n_trees,
                 'max_iter': tree.stopper.max_iter,
                 'cross_prob': tree.crosser.cross_prob,
                 'mutation_prob': tree.mutator.mutation_prob,
                 ###'initialization': tree.initializer.initialization,
                 ###'metric': tree.evaluator.metric,
                 ###'selection': tree.selector.selection,
                 'n_elitism': tree.selector.n_elitism,
                 'early_stopping': tree.stopper.early_stopping,
                 'n_iter_no_change': tree.stopper.n_iter_no_change,

                 # additional genetic algorithm params
                 'cross_both': tree.crosser.cross_both,
                 ###'mutations_additional': tree.mutator.mutations_additional,
                 'mutation_replace': tree.mutator.mutation_replace,
                 'initial_depth': tree.initializer.initial_depth,
                 'split_prob': tree.initializer.split_prob,
                 ###'n_leaves_factor': tree.n_leaves_factor,
                 ###'depth_factor': tree.depth_factor,
                 ###'tournament_size': tree.tournament_size,
                 ###'leave_selected_parents': tree.leave_selected_parents,

                 # technical params
                 'random_state': config['computation']['random_seed'],  
    
    }
    
    


class CustomProgressReporter(JupyterNotebookReporter):
    def report(self, trials, *args, **kwargs):
        metrics = kwargs.get("metrics", None)
        if metrics is not None:
            metrics_to_show = self._metric_columns
            output_str = self._make_output_str(
                trials, metrics, metrics_to_show)
            self._display(output_str)

            
def get_datasets_from_dataset_dict(dataset_dict, with_valid=True, encoded=True, cv=False):         
        
    if not cv:
        if with_valid:
            if encoded:
                X_train = dataset_dict['X_train']
                X_valid = dataset_dict['X_valid']
                X_test = dataset_dict['X_test']

                y_train = dataset_dict['y_train']
                y_valid = dataset_dict['y_valid']
                y_test = dataset_dict['y_test']          

            else:
                X_train = dataset_dict['X_train_raw']
                X_valid = dataset_dict['X_valid_raw']
                X_test = dataset_dict['X_test_raw']

                y_train = dataset_dict['y_train_raw']
                y_valid = dataset_dict['y_valid_raw']
                y_test = dataset_dict['y_test_raw']          
        else:
            if encoded:
                X_train = dataset_dict['X_train_no_valid']
                X_valid = None
                X_test = dataset_dict['X_test_no_valid']

                y_train = dataset_dict['y_train_no_valid']
                y_valid = None
                y_test = dataset_dict['y_test_no_valid']           
            else:
                X_train = dataset_dict['X_train_raw_no_valid']
                X_valid = None
                X_test = dataset_dict['X_test_raw_no_valid']

                y_train = dataset_dict['y_train_raw_no_valid']
                y_valid = None
                y_test = dataset_dict['y_test_raw_no_valid']    
    else:
        if with_valid:
            if encoded:
                X_train = dataset_dict['X_train_cv']
                X_valid = dataset_dict['X_valid_cv']
                X_test = dataset_dict['X_test_cv']

                y_train = dataset_dict['y_train_cv']
                y_valid = dataset_dict['y_valid_cv']
                y_test = dataset_dict['y_test_cv']          

            else:
                X_train = dataset_dict['X_train_raw_cv']
                X_valid = dataset_dict['X_valid_raw_cv']
                X_test = dataset_dict['X_test_raw_cv']

                y_train = dataset_dict['y_train_raw_cv']
                y_valid = dataset_dict['y_valid_raw_cv']
                y_test = dataset_dict['y_test_raw_cv']          
        else:
            if encoded:
                X_train = dataset_dict['X_train_no_valid_cv']
                X_valid = None
                X_test = dataset_dict['X_test_no_valid_cv']

                y_train = dataset_dict['y_train_no_valid_cv']
                y_valid = None
                y_test = dataset_dict['y_test_no_valid_cv']           
            else:
                X_train = dataset_dict['X_train_raw_no_valid_cv']
                X_valid = None
                X_test = dataset_dict['X_test_raw_no_valid_cv']

                y_train = dataset_dict['y_train_raw_no_valid_cv']
                y_valid = None
                y_test = dataset_dict['y_test_raw_no_valid_cv']          
        
    
    return (X_train, y_train), (X_valid, y_valid),(X_test, y_test)
    
    
def hpo_GRANDE(identifier, dataset_dict_list, parameter_dict, config, metric='f1', greater_better=True, timestr = 'placeholder'):
    def evaluate_parameter_setting_GRANDE(trial_config):

        parameter_setting = trial_config['parameters']
        dataset_dict_list = ray.get(trial_config['dataset_dict_list'])
        config = trial_config['config'] 

        config['GRANDE'].update(parameter_setting)
        
        with_valid = True
        encoded = True         

        score_base_model_list = []
        runtime_list = []     

        for i in range(config['computation']['cv_num_eval']):   

            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_cv_list = []
                runtime_list_cv = []                  
                for j in range(config['computation']['cv_num_hpo']):

                    dataset_dict = dataset_dict_list[i][j]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded, cv=True)

                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

                    start = timeit.default_timer()
                    
                    dataset_dict_updated = {'X_train': X_train,
                                            'y_train': y_train,
                                            'X_valid': X_valid,
                                            'y_valid': y_valid,
                                            'X_test': X_test,
                                            'y_test': y_test,
                                           }
                    for restart_number in range(config['GRANDE']['restarts']+1):
            
                        tf.keras.backend.clear_session()
                             
                        best_model = None
                        best_score = np.inf if config['GRANDE']['restart_type'] == 'loss' else -np.inf

                        (train_data, 
                         valid_data, 
                         batch_size, 
                         batch_size_val, 
                         class_weight_dict, 
                         loss_function, 
                         optimizer_function_dict,
                         metrics_GRANDE, 
                         callbacks) = pepare_GRANDE_for_training(config, dataset_dict_updated, number_of_classes, timestr)
            
                        config['GRANDE']['number_of_classes'] = number_of_classes
                        config['GRANDE']['number_of_variables'] = dataset_dict_updated['X_train'].shape[1]
                        config['GRANDE']['mean'] = np.mean(dataset_dict_updated['y_train'])
                        config['GRANDE']['std'] = np.std(dataset_dict_updated['y_train'])
                        config['GRANDE']['random_seed'] = config['computation']['random_seed']
            
                        model = GRANDE(**config['GRANDE'])  
            
                        #model.set_params(**config['GRANDE'])   
                        model.weights_optimizer = optimizer_function_dict['weights_optimizer']
                        model.index_optimizer = optimizer_function_dict['index_optimizer']
                        model.values_optimizer = optimizer_function_dict['values_optimizer']
                        model.leaf_optimizer = optimizer_function_dict['leaf_optimizer']            
            
                        model.compile(loss=loss_function, metrics=metrics_GRANDE)
            
                        history = model.fit(train_data,
                                              epochs=config['GRANDE']['epochs'], 
                                              #steps_per_epoch=dataset_dict_updated['X_train'].shape[0]//64,
                                              validation_data=valid_data,
                                              callbacks=callbacks,
                                              class_weight = class_weight_dict,
                                              verbose=0) 
            
                        best_score_current_model = min(history.history['val_loss']) if config['GRANDE']['restart_type'] == 'loss' else max(history.history['val_' + metrics_GRANDE[0].name])              
                        model_params = model.get_params()
            
                        if (best_score_current_model < best_score and  config['GRANDE']['restart_type'] == 'loss') or (best_score_current_model > best_score and  config['GRANDE']['restart_type'] == 'metric'):
                            best_score = best_score_current_model
                            best_model = model
            

                    base_model = best_model      
              
                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred_proba = tf.squeeze(base_model.predict(enforce_numpy(X_test, dtype='float'), batch_size=batch_size, verbose=0)) if number_of_classes <= 2 else base_model.predict(enforce_numpy(X_test, dtype='float'), batch_size=batch_size, verbose=0)
                    base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                    base_model_pred = np.round(base_model_pred_proba)
                    if config['GRANDE']['objective'] == 'classification':
                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_cv = 0.5

                    else:
                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_cv_list.append(score_base_model_cv)
                    runtime_list_cv.append(runtime)

                score_base_model = np.mean(score_base_model_cv_list)

                score_base_model_list.append(score_base_model)
                runtime_list.append(np.mean(runtime_list_cv))
            else:

                dataset_dict = dataset_dict_list[i]
                
                ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))      

                start = timeit.default_timer()

                dataset_dict_updated = {'X_train': X_train,
                                        'y_train': y_train,
                                        'X_valid': X_valid,
                                        'y_valid': y_valid,
                                        'X_test': X_test,
                                        'y_test': y_test,
                                       }
                for restart_number in range(config['GRANDE']['restarts']+1):
        
                    tf.keras.backend.clear_session()
                         
                    best_model = None
                    best_score = np.inf if config['GRANDE']['restart_type'] == 'loss' else -np.inf

                    (train_data, 
                     valid_data, 
                     batch_size, 
                     batch_size_val, 
                     class_weight_dict, 
                     loss_function, 
                     optimizer_function_dict,
                     metrics_GRANDE, 
                     callbacks) = pepare_GRANDE_for_training(config, dataset_dict_updated, number_of_classes, timestr)
        
                    config['GRANDE']['number_of_classes'] = number_of_classes
                    config['GRANDE']['number_of_variables'] = dataset_dict_updated['X_train'].shape[1]
                    config['GRANDE']['mean'] = np.mean(dataset_dict_updated['y_train'])
                    config['GRANDE']['std'] = np.std(dataset_dict_updated['y_train'])
                    config['GRANDE']['random_seed'] = config['computation']['random_seed']
        
                    model = GRANDE(**config['GRANDE'])  
        
                    #model.set_params(**config['GRANDE'])   
                    model.weights_optimizer = optimizer_function_dict['weights_optimizer']
                    model.index_optimizer = optimizer_function_dict['index_optimizer']
                    model.values_optimizer = optimizer_function_dict['values_optimizer']
                    model.leaf_optimizer = optimizer_function_dict['leaf_optimizer']            
        
                    model.compile(loss=loss_function, metrics=metrics_GRANDE)
        
                    history = model.fit(train_data,
                                          epochs=config['GRANDE']['epochs'], 
                                          #steps_per_epoch=dataset_dict_updated['X_train'].shape[0]//64,
                                          validation_data=valid_data,
                                          callbacks=callbacks,
                                          class_weight = class_weight_dict,
                                          verbose=0) 
        
                    best_score_current_model = min(history.history['val_loss']) if config['GRANDE']['restart_type'] == 'loss' else max(history.history['val_' + metrics_GRANDE[0].name])              
                    model_params = model.get_params()
        
                    if (best_score_current_model < best_score and  config['GRANDE']['restart_type'] == 'loss') or (best_score_current_model > best_score and  config['GRANDE']['restart_type'] == 'metric'):
                        best_score = best_score_current_model
                        best_model = model
        

                base_model = best_model             

                end = timeit.default_timer()  
                runtime = end - start                

                base_model_pred_proba = tf.squeeze(base_model.predict(enforce_numpy(X_test, dtype='float'), batch_size=batch_size, verbose=0)) if number_of_classes <= 2 else base_model.predict(enforce_numpy(X_test, dtype='float'), batch_size=batch_size, verbose=0)
                base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                base_model_pred = np.round(base_model_pred_proba)
                if config['GRANDE']['objective'] == 'classification':

                    if metric not in ['f1', 'roc_auc', 'accuracy']:
                        score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))
                    else:
                        if metric == 'f1':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                        if metric == 'accuracy':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                                
                        elif metric == 'roc_auc':
                            try:
                                if number_of_classes > 2:                            
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                else:
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                            except ValueError:
                                score_base_model = 0.5
                else:
                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                score_base_model_list.append(score_base_model)
                runtime_list.append(runtime)      


        if config['computation']['report_test_performance']:
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_test_list = []
                runtime_test_list = []                       

                for i in range(config['computation']['cv_num_eval']):   

                    dataset_dict = dataset_dict_list[i][0]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))              

                    start = timeit.default_timer()

                    dataset_dict_updated = {'X_train': X_train,
                                            'y_train': y_train,
                                            'X_valid': X_valid,
                                            'y_valid': y_valid,
                                            'X_test': X_test,
                                            'y_test': y_test,
                                           }
                    for restart_number in range(config['GRANDE']['restarts']+1):
            
                        tf.keras.backend.clear_session()
                             
                        best_model = None
                        best_score = np.inf if config['GRANDE']['restart_type'] == 'loss' else -np.inf

                        (train_data, 
                         valid_data, 
                         batch_size, 
                         batch_size_val, 
                         class_weight_dict, 
                         loss_function, 
                         optimizer_function_dict,
                         metrics_GRANDE, 
                         callbacks) = pepare_GRANDE_for_training(config, dataset_dict_updated, number_of_classes, timestr)
            
                        config['GRANDE']['number_of_classes'] = number_of_classes
                        config['GRANDE']['number_of_variables'] = dataset_dict_updated['X_train'].shape[1]
                        config['GRANDE']['mean'] = np.mean(dataset_dict_updated['y_train'])
                        config['GRANDE']['std'] = np.std(dataset_dict_updated['y_train'])
                        config['GRANDE']['random_seed'] = config['computation']['random_seed']
            
                        model = GRANDE(**config['GRANDE'])  
            
                        #model.set_params(**config['GRANDE'])   
                        model.weights_optimizer = optimizer_function_dict['weights_optimizer']
                        model.index_optimizer = optimizer_function_dict['index_optimizer']
                        model.values_optimizer = optimizer_function_dict['values_optimizer']
                        model.leaf_optimizer = optimizer_function_dict['leaf_optimizer']            
            
                        model.compile(loss=loss_function, metrics=metrics_GRANDE)
            
                        history = model.fit(train_data,
                                              epochs=config['GRANDE']['epochs'], 
                                              #steps_per_epoch=dataset_dict_updated['X_train'].shape[0]//64,
                                              validation_data=valid_data,
                                              callbacks=callbacks,
                                              class_weight = class_weight_dict,
                                              verbose=0) 
            
                        best_score_current_model = min(history.history['val_loss']) if config['GRANDE']['restart_type'] == 'loss' else max(history.history['val_' + metrics_GRANDE[0].name])              
                        model_params = model.get_params()
            
                        if (best_score_current_model < best_score and  config['GRANDE']['restart_type'] == 'loss') or (best_score_current_model > best_score and  config['GRANDE']['restart_type'] == 'metric'):
                            best_score = best_score_current_model
                            best_model = model
            

                    base_model = best_model   

                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred_proba = tf.squeeze(base_model.predict(enforce_numpy(X_test, dtype='float'), batch_size=batch_size, verbose=0)) if number_of_classes <= 2 else base_model.predict(enforce_numpy(X_test, dtype='float'), batch_size=batch_size, verbose=0)
                    base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                    base_model_pred = np.round(base_model_pred_proba)

                    if config['GRANDE']['objective'] == 'classification':

                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_test = 0.5

                    else:
                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_test_list.append(score_base_model_test)
                    runtime_test_list.append(runtime)

            else:
                score_base_model_test_list = score_base_model_list
        else:
            score_base_model_test_list = score_base_model_list                

        return {
            'cv_score': np.mean(score_base_model_list),
            'test_score': np.mean(score_base_model_test_list),
            'runtime': np.mean(runtime_list),
            'parameter_setting': parameter_setting,
        }

   
    if config['computation']['verbose'] > 0:
        print('Number of Trials GRANDE: ' + str(config['computation']['search_iterations']))
    
    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
        hpo_mode = "max" if metric in ['r2', 'accuracy', 'f1', 'roc_auc'] else "min"

        initial_params_update = [
            {
                    'num_layers': 1,
                    'dense_layer_identifier': [],
                
                    'depth': 6,
                    'n_estimators': 2048,
                    
                    'learning_rate_weights': 0.01,
                    'learning_rate_index': 0.01,
                    'learning_rate_values': 0.01,
                    'learning_rate_leaf': 0.01,
            
                    'loss': 'crossentropy',
                
                    'dropout': 0.00,
                    'optimizer': 'SWA',
            },
            {
                    'num_layers': 1,
                    'dense_layer_identifier': [],
                
                    'depth': 6,
                    'n_estimators': 2048,
                    
                    'learning_rate_weights': 0.01,
                    'learning_rate_index': 0.01,
                    'learning_rate_values': 0.01,
                    'learning_rate_leaf': 0.01,
            
                    'loss': 'focal_crossentropy',
                
                    'dropout': 0.00,
                    'optimizer': 'SWA',
            },            
        ]    
        
        
        initial_params = [{key: value.sample() for key, value in parameter_dict.items()}, 
                          {key: value.sample() for key, value in parameter_dict.items()}]
        initial_params[0].update(initial_params_update[0])
        initial_params[1].update(initial_params_update[1])

        optuna_search = OptunaSearch(metric="cv_score", mode=hpo_mode, points_to_evaluate=initial_params, seed=config['computation']['random_seed'])

        config_ray = {
                      'parameters': parameter_dict,
                      'dataset_dict_list': dataset_dict_list, 
                      'config': config,
                     }

        print_intermediate_tables = True if config['computation']['verbose'] > 2 else False 
        progress_reporter = JupyterNotebookReporter(overwrite=False,metric_columns=['cv_score','test_score','runtime'],max_progress_rows=config['computation']['search_iterations'],print_intermediate_tables=False,max_column_length=50) if config['computation']['verbose'] > 0 else JupyterNotebookReporter(overwrite=False, max_progress_rows=0, metric_columns=[], parameter_columns=[])

        analysis = tune.run(
                            evaluate_parameter_setting_GRANDE,
                            config=config_ray,
                            search_alg=optuna_search,
                            resources_per_trial={"cpu": config['computation']['n_jobs'], 'gpu': config['computation']['num_gpu']},
                            num_samples=config['computation']['search_iterations'],
                            #log_to_file=False,
                            local_dir='./results_ray/' + timestr + '/',
                            #callbacks=[],
                            raise_on_failed_trial=False,        
                            verbose=config['computation']['verbose'],
                            progress_reporter=progress_reporter, #,parameter_columns=['cv_score','test_score','runtime']        
                            #progress_reporter=CustomProgressReporter(metric_columns=['cv_score','test_score','runtime']),
                            #keep_checkpoints_num=1,
                            #checkpoint_score_attr='cv_score'
                           )    

    grid_search_results = [tuple(trial.last_result.values()) for trial in analysis.trials]          
           
    replace_value = -np.inf if greater_better else np.inf

    replace_dict = {}
    for some_dict in [result[3] for result in grid_search_results]:
        if isinstance(some_dict, dict):
            for key in some_dict.keys():
                replace_dict[key] = None 
            break

    grid_search_results = [[np.nan_to_num(pd.to_numeric(result[0], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[1], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[2], errors='coerce'), nan=replace_value), 
                            result[3] if isinstance(result[3], dict) else replace_dict] for result in grid_search_results]
    
    cv_scores_base_model = np.nan_to_num([result[0] for result in grid_search_results], nan=replace_value)
    test_scores_base_model = np.nan_to_num([result[1] for result in grid_search_results], nan=replace_value)
    runtime_base_model = np.nan_to_num([result[2] for result in grid_search_results], nan=np.inf)
    parameter_settings = [result[3] for result in grid_search_results]

    grid_search_results_sorted = sorted(grid_search_results, key=lambda tup: tup[0], reverse=greater_better) 

    best_score_cv = grid_search_results_sorted[0][0]
    best_score_test = grid_search_results_sorted[0][1]
    best_model_runtime_cv = grid_search_results_sorted[0][2]
    best_params = grid_search_results_sorted[0][3]
       
    model_identifier = 'GRANDE'
            
    if config['computation']['report_test_performance']:
        with_valid = True
        encoded = True                       
        
        runtime_list = []
        base_model_list = []           
        dataset_dict_list = ray.get(dataset_dict_list)
        for i in range(config['computation']['cv_num_eval']):
            try:
                dataset_dict = dataset_dict_list[i][0]
            except:
                dataset_dict = dataset_dict_list[i]
                
            ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

            number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

            start = timeit.default_timer()

            dataset_dict_updated = {'X_train': X_train,
                                    'y_train': y_train,
                                    'X_valid': X_valid,
                                    'y_valid': y_valid,
                                    'X_test': X_test,
                                    'y_test': y_test,
                                   }
            for restart_number in range(config['GRANDE']['restarts']+1):
    
                tf.keras.backend.clear_session()
                     
                best_model = None
                best_score = np.inf if config['GRANDE']['restart_type'] == 'loss' else -np.inf

                (train_data, 
                 valid_data, 
                 batch_size, 
                 batch_size_val, 
                 class_weight_dict, 
                 loss_function, 
                 optimizer_function_dict,
                 metrics_GRANDE, 
                 callbacks) = pepare_GRANDE_for_training(config, dataset_dict_updated, number_of_classes, timestr)
    
                config['GRANDE']['number_of_classes'] = number_of_classes
                config['GRANDE']['number_of_variables'] = dataset_dict_updated['X_train'].shape[1]
                config['GRANDE']['mean'] = np.mean(dataset_dict_updated['y_train'])
                config['GRANDE']['std'] = np.std(dataset_dict_updated['y_train'])
                config['GRANDE']['random_seed'] = config['computation']['random_seed']
    
                model = GRANDE(**config['GRANDE'])  
    
                #model.set_params(**config['GRANDE'])   
                model.weights_optimizer = optimizer_function_dict['weights_optimizer']
                model.index_optimizer = optimizer_function_dict['index_optimizer']
                model.values_optimizer = optimizer_function_dict['values_optimizer']
                model.leaf_optimizer = optimizer_function_dict['leaf_optimizer']            
    
                model.compile(loss=loss_function, metrics=metrics_GRANDE)
    
                history = model.fit(train_data,
                                      epochs=config['GRANDE']['epochs'], 
                                      #steps_per_epoch=dataset_dict_updated['X_train'].shape[0]//64,
                                      validation_data=valid_data,
                                      callbacks=callbacks,
                                      class_weight = class_weight_dict,
                                      verbose=0) 
    
                best_score_current_model = min(history.history['val_loss']) if config['GRANDE']['restart_type'] == 'loss' else max(history.history['val_' + metrics_GRANDE[0].name])              
                model_params = model.get_params()
    
                if (best_score_current_model < best_score and  config['GRANDE']['restart_type'] == 'loss') or (best_score_current_model > best_score and  config['GRANDE']['restart_type'] == 'metric'):
                    best_score = best_score_current_model
                    best_model = model
    

            base_model = best_model               
            
            end = timeit.default_timer()  
            runtime = end - start
            
            runtime_list.append(runtime)
            base_model_list.append(base_model)
        
    else:
        runtime_list = [best_model_runtime_cv for _ in range(config['computation']['cv_num_eval'])]
        base_model_list = [None for _ in range(config['computation']['cv_num_eval'])]
    
    if config['computation']['verbose'] > 0:
        print('Best Params GRANDE')
        display(best_params)        
        print('Best CV-Score GRANDE:', best_score_cv)

    hpo_path_by_dataset =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/' + identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/').mkdir(parents=True, exist_ok=True)    
        
    if not os.path.isfile(hpo_path_by_dataset):
        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list_one_level(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list_one_level([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()        
    
    hpo_path_by_dataset_with_timestamp =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/' + model_identifier + '___' +  identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/').mkdir(parents=True, exist_ok=True)    
    
    if not os.path.isfile(hpo_path_by_dataset_with_timestamp):
        file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list_one_level(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list_one_level([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()           
    
    return (best_score_cv, runtime_list, base_model_list), cv_scores_base_model, test_scores_base_model
   
    
    
    
    
def hpo_CatBoost(identifier, dataset_dict_list, parameter_dict, config, metric='f1', greater_better=True, timestr = 'placeholder'):

    def evaluate_parameter_setting_CatBoost(trial_config):   
              
        parameter_setting = trial_config['parameters']
        dataset_dict_list = ray.get(trial_config['dataset_dict_list'])
        config = trial_config['config']
        
        with_valid = True
        encoded = False if config['preprocessing']['CatBoostEncoding'] else True        
        
        score_base_model_list = []
        runtime_list = []     
        
        if config['GRANDE']['objective'] == 'classification':
             model_class = CatBoostClassifier
        else:
             model_class = CatBoostRegressor  
                            
        for i in range(config['computation']['cv_num_eval']):   
            
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_cv_list = []
                runtime_list_cv = []                  
                for j in range(config['computation']['cv_num_hpo']):

                    dataset_dict = dataset_dict_list[i][j]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded, cv=True)
                    
                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

                    cat_features = dataset_dict['categorical_feature_indices'] if config['preprocessing']['CatBoostEncoding'] else None

                    if config['GRANDE']['objective'] == 'classification':
                        if config['GRANDE']['class_weights']:
                            train_data = Pool(
                                    data=enforce_numpy(X_train),
                                    label=enforce_numpy(y_train),
                                    cat_features=cat_features,
                                    weight=calculate_sample_weights(enforce_numpy(y_train))
                                )

                            eval_data = Pool(
                                    data=enforce_numpy(X_valid),
                                    label=enforce_numpy(y_valid),
                                    cat_features=cat_features,
                                    weight=calculate_sample_weights(enforce_numpy(y_valid))
                                ) 
                        else:
                            train_data = Pool(
                                    data=enforce_numpy(X_train),
                                    label=enforce_numpy(y_train),
                                    cat_features=cat_features,
                                )

                            eval_data = Pool(
                                    data=enforce_numpy(X_valid),
                                    label=enforce_numpy(y_valid),
                                    cat_features=cat_features,
                                )                                 

                    else:
                        train_data = Pool(
                                data=enforce_numpy(X_train),
                                label=enforce_numpy(y_train),
                                cat_features=cat_features,
                            )

                        eval_data = Pool(
                                data=enforce_numpy(X_valid),
                                label=enforce_numpy(y_valid),
                                cat_features=cat_features,
                            )                      
                    
                    base_model = model_class()
                    
                    
                    base_model.set_params(**parameter_setting)

                    start = timeit.default_timer()
                    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                        base_model.fit(X=train_data, eval_set=eval_data)

                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)
                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_cv = 0.5
                    
                    else:
                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_cv_list.append(score_base_model_cv)
                    runtime_list_cv.append(runtime)

                score_base_model = np.mean(score_base_model_cv_list)

                score_base_model_list.append(score_base_model)
                runtime_list.append(np.mean(runtime_list_cv))
            else:

                dataset_dict = dataset_dict_list[i]
                
                ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))
                cat_features = dataset_dict['categorical_feature_indices'] if config['preprocessing']['CatBoostEncoding'] else None

                if config['GRANDE']['objective'] == 'classification':
                    if config['GRANDE']['class_weights']:
                        train_data = Pool(
                                data=enforce_numpy(X_train),
                                label=enforce_numpy(y_train),
                                cat_features=cat_features,
                                weight=calculate_sample_weights(enforce_numpy(y_train))
                            )

                        eval_data = Pool(
                                data=enforce_numpy(X_valid),
                                label=enforce_numpy(y_valid),
                                cat_features=cat_features,
                                weight=calculate_sample_weights(enforce_numpy(y_valid))
                            ) 
                    else:
                        train_data = Pool(
                                data=enforce_numpy(X_train),
                                label=enforce_numpy(y_train),
                                cat_features=cat_features,
                            )

                        eval_data = Pool(
                                data=enforce_numpy(X_valid),
                                label=enforce_numpy(y_valid),
                                cat_features=cat_features,
                            )                                 

                else:
                    train_data = Pool(
                            data=enforce_numpy(X_train),
                            label=enforce_numpy(y_train),
                            cat_features=cat_features,
                        )

                    eval_data = Pool(
                            data=enforce_numpy(X_valid),
                            label=enforce_numpy(y_valid),
                            cat_features=cat_features,
                        )                     

                base_model = model_class()

                base_model.set_params(**parameter_setting)

                start = timeit.default_timer()
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
                        base_model.fit(X=train_data, eval_set=eval_data)

                end = timeit.default_timer()  
                runtime = end - start                     

                base_model_pred = base_model.predict(enforce_numpy(X_test))
                base_model_pred = np.nan_to_num(base_model_pred)
                if config['GRANDE']['objective'] == 'classification':
                    base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                    base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                    if metric not in ['f1', 'roc_auc', 'accuracy']:
                        score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))
                    else:
                        if metric == 'f1':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                        if metric == 'accuracy':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                                
                        elif metric == 'roc_auc':
                            try:
                                if number_of_classes > 2:                            
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                else:
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                            except ValueError:
                                score_base_model = 0.5
                else:
                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                score_base_model_list.append(score_base_model)
                runtime_list.append(runtime)                
                
        
        if config['computation']['report_test_performance']:
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_test_raw_list = []
                runtime_test_raw_list = []                       

                for i in range(config['computation']['cv_num_eval']):   

                    dataset_dict = dataset_dict_list[i][0]

                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)
                    
                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))
                    cat_features = dataset_dict['categorical_feature_indices'] if config['preprocessing']['CatBoostEncoding'] else None

                    if config['GRANDE']['objective'] == 'classification':
                        if config['GRANDE']['class_weights']:
                            train_data = Pool(
                                    data=enforce_numpy(X_train),
                                    label=enforce_numpy(y_train),
                                    cat_features=cat_features,
                                    weight=calculate_sample_weights(enforce_numpy(y_train))
                                )

                            eval_data = Pool(
                                    data=enforce_numpy(X_valid),
                                    label=enforce_numpy(y_valid),
                                    cat_features=cat_features,
                                    weight=calculate_sample_weights(enforce_numpy(y_valid))
                                ) 
                        else:
                            train_data = Pool(
                                    data=enforce_numpy(X_train),
                                    label=enforce_numpy(y_train),
                                    cat_features=cat_features,
                                )

                            eval_data = Pool(
                                    data=enforce_numpy(X_valid),
                                    label=enforce_numpy(y_valid),
                                    cat_features=cat_features,
                                )                                 

                    else:
                        train_data = Pool(
                                data=enforce_numpy(X_train),
                                label=enforce_numpy(y_train),
                                cat_features=cat_features,
                            )

                        eval_data = Pool(
                                data=enforce_numpy(X_valid),
                                label=enforce_numpy(y_valid),
                                cat_features=cat_features,
                            )                      
                    
                    base_model = model_class()
                    
                    
                    base_model.set_params(**parameter_setting)

                    start = timeit.default_timer()
                    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                        with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
                            base_model.fit(X=train_data, eval_set=eval_data)

                    end = timeit.default_timer()  
                    runtime = end - start               

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)
                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_test_raw = 0.5

                    else:
                        score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_test_raw_list.append(score_base_model_test_raw)
                    runtime_test_raw_list.append(runtime)
      
            else:
                score_base_model_test_raw_list = score_base_model_list
        else:
            score_base_model_test_raw_list = score_base_model_list                
        
        return {
            'cv_score': np.mean(score_base_model_list),
            'test_score': np.mean(score_base_model_test_raw_list),
            'runtime': np.mean(runtime_list),
            'parameter_setting': parameter_setting,
        }


    if config['computation']['verbose'] > 0:
        print('Number of Trials CatBoost: ' + str(config['computation']['search_iterations']))
    
    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
        hpo_mode = "max" if metric in ['r2', 'accuracy', 'f1', 'roc_auc'] else "min"

        initial_params_update = [{
                'learning_rate': 0.1,#TabSurvey
                'max_depth': 6,#TabSurvey
                'l2_leaf_reg': 3,#TabSurvey
        }]   

        initial_params = [{key: value.sample() for key, value in parameter_dict.items()}]
        initial_params[0].update(initial_params_update[0])
        
        optuna_search = OptunaSearch(metric="cv_score", mode=hpo_mode, points_to_evaluate=initial_params, seed=config['computation']['random_seed'])

        config_ray = {
                      'parameters': parameter_dict,
                      'dataset_dict_list': dataset_dict_list, 
                      'config': config,
                     }

        print_intermediate_tables = True if config['computation']['verbose'] > 2 else False 
        progress_reporter = JupyterNotebookReporter(overwrite=False,metric_columns=['cv_score','test_score','runtime'],max_progress_rows=config['computation']['search_iterations'],print_intermediate_tables=False,max_column_length=50) if config['computation']['verbose'] > 0 else JupyterNotebookReporter(overwrite=False, max_progress_rows=0, metric_columns=[], parameter_columns=[])

        analysis = tune.run(
                            evaluate_parameter_setting_CatBoost,
                            config=config_ray,
                            search_alg=optuna_search,
                            resources_per_trial={"cpu": config['computation']['n_jobs'], 'gpu': config['computation']['num_gpu']},
                            num_samples=config['computation']['search_iterations'],
                            #log_to_file=False,
                            local_dir='./results_ray/' + timestr + '/',
                            #callbacks=[],
                            raise_on_failed_trial=False,        
                            verbose=config['computation']['verbose'],
                            progress_reporter=progress_reporter, #,parameter_columns=['cv_score','test_score','runtime']        
                            #progress_reporter=CustomProgressReporter(metric_columns=['cv_score','test_score','runtime']),
                            #keep_checkpoints_num=1,
                            #checkpoint_score_attr='cv_score'
                           )    

    grid_search_results = [tuple(trial.last_result.values()) for trial in analysis.trials]          
           
    replace_value = -np.inf if greater_better else np.inf

    replace_dict = {}
    for some_dict in [result[3] for result in grid_search_results]:
        if isinstance(some_dict, dict):
            for key in some_dict.keys():
                replace_dict[key] = None 
            break

    grid_search_results = [[np.nan_to_num(pd.to_numeric(result[0], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[1], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[2], errors='coerce'), nan=replace_value), 
                            result[3] if isinstance(result[3], dict) else replace_dict] for result in grid_search_results]
    
    cv_scores_base_model = np.nan_to_num([result[0] for result in grid_search_results], nan=replace_value)
    test_scores_base_model = np.nan_to_num([result[1] for result in grid_search_results], nan=replace_value)
    runtime_base_model = np.nan_to_num([result[2] for result in grid_search_results], nan=np.inf)
    parameter_settings = [result[3] for result in grid_search_results]


    grid_search_results_sorted = sorted(grid_search_results, key=lambda tup: tup[0], reverse=greater_better) 

    best_score_cv = grid_search_results_sorted[0][0]
    best_score_test_raw = grid_search_results_sorted[0][1]
    best_model_runtime_cv = grid_search_results_sorted[0][2]
    best_params = grid_search_results_sorted[0][3]
   
    if config['GRANDE']['objective'] == 'classification':
        model_class = CatBoostClassifier
        model_identifier = 'CatBoost_class'
    else:
        model_class = CatBoostRegressor      
        model_identifier = 'CatBoost_reg'
            
    if config['computation']['report_test_performance']:
        with_valid = True
        encoded = False if config['preprocessing']['CatBoostEncoding'] else True
        
        runtime_list = []
        base_model_list = []           
        dataset_dict_list = ray.get(dataset_dict_list)
        for i in range(config['computation']['cv_num_eval']):
            try:
                dataset_dict = dataset_dict_list[i][0]
            except:
                dataset_dict = dataset_dict_list[i]
                
            ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

            number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

            cat_features = dataset_dict['categorical_feature_indices'] if config['preprocessing']['CatBoostEncoding'] else None

            if config['GRANDE']['objective'] == 'classification':
                if config['GRANDE']['class_weights']:
                    train_data = Pool(
                            data=enforce_numpy(X_train),
                            label=enforce_numpy(y_train),
                            cat_features=cat_features,
                            weight=calculate_sample_weights(enforce_numpy(y_train))
                        )

                    eval_data = Pool(
                            data=enforce_numpy(X_valid),
                            label=enforce_numpy(y_valid),
                            cat_features=cat_features,
                            weight=calculate_sample_weights(enforce_numpy(y_valid))
                        ) 
                else:
                    train_data = Pool(
                            data=enforce_numpy(X_train),
                            label=enforce_numpy(y_train),
                            cat_features=cat_features,
                        )

                    eval_data = Pool(
                            data=enforce_numpy(X_valid),
                            label=enforce_numpy(y_valid),
                            cat_features=cat_features,
                        )                                 

            else:
                train_data = Pool(
                        data=enforce_numpy(X_train),
                        label=enforce_numpy(y_train),
                        cat_features=cat_features,
                    )

                eval_data = Pool(
                        data=enforce_numpy(X_valid),
                        label=enforce_numpy(y_valid),
                        cat_features=cat_features,
                    )                      

            base_model = model_class()


            base_model.set_params(**best_params)

            start = timeit.default_timer()
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
                    base_model.fit(X=train_data, eval_set=eval_data)

            end = timeit.default_timer()  
            runtime = end - start       

            runtime_list.append(runtime)
            base_model_list.append(base_model)
        
    else:
        runtime_list = [best_model_runtime_cv for _ in range(config['computation']['cv_num_eval'])]
        base_model_list = [None for _ in range(config['computation']['cv_num_eval'])]
    
    if config['computation']['verbose'] > 0:
        print('Best Params CatBoost')
        display(best_params)        
        print('Best CV-Score CatBoost:', best_score_cv)

    hpo_path_by_dataset =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/' + identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/').mkdir(parents=True, exist_ok=True)    
        
    if not os.path.isfile(hpo_path_by_dataset):
        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()        
    
    hpo_path_by_dataset_with_timestamp =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/' + model_identifier + '___' +  identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/').mkdir(parents=True, exist_ok=True)    
    
    if not os.path.isfile(hpo_path_by_dataset_with_timestamp):
        file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()        
    
    return (best_score_cv, runtime_list, base_model_list), cv_scores_base_model, test_scores_base_model
  
        
def hpo_XGB(identifier, dataset_dict_list, parameter_dict, config, metric='f1', greater_better=True, timestr = 'placeholder'):
    
    def evaluate_parameter_setting_XGB(trial_config):   
        
        parameter_setting = trial_config['parameters']
        dataset_dict_list = ray.get(trial_config['dataset_dict_list'])
        config = trial_config['config']
        
        with_valid = True
        encoded = False if config['preprocessing']['XGBoostEncoding'] else True
        
        score_base_model_list = []
        runtime_list = []     
        
        if config['GRANDE']['objective'] == 'classification':
             model_class = XGBClassifier
        else:
             model_class = XGBRegressor  
                            
        for i in range(config['computation']['cv_num_eval']):   
            
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_cv_list = []
                runtime_list_cv = []                  
                for j in range(config['computation']['cv_num_hpo']):
                    
                    dataset_dict = dataset_dict_list[i][j]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded, cv=True)
                    
                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

                    if config['preprocessing']['XGBoostEncoding']:
                        feature_types = []
                        for feature_index in range(X_train.shape[1]):
                            if feature_index in dataset_dict['categorical_feature_indices']:
                                feature_types.append('c')
                            else:
                                feature_types.append('q')

                        base_model = model_class(enable_categorical=True, feature_types=feature_types)                      
                    else:
                        base_model = model_class()                      

                    base_model.set_params(**parameter_setting)

                    start = timeit.default_timer()
                    if config['GRANDE']['class_weights']:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train),
                                       sample_weight=calculate_sample_weights(enforce_numpy(y_train)),
                                       eval_set=[(enforce_numpy(X_valid), enforce_numpy(y_valid))], 
                                       sample_weight_eval_set=[calculate_sample_weights(enforce_numpy(y_valid))],  
                                       #eval_metric='auc' if config['GRANDE']['objective'] == 'classification' else 'mae',
                                       verbose=False)
                    else:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train),
                                       eval_set=[(enforce_numpy(X_valid), enforce_numpy(y_valid))], 
                                       #eval_metric='auc' if config['GRANDE']['objective'] == 'classification' else 'mae',
                                       verbose=False)                        

                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)
                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_cv = 0.5
                    
                    else:
                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_cv_list.append(score_base_model_cv)
                    runtime_list_cv.append(runtime)

                score_base_model = np.mean(score_base_model_cv_list)

                score_base_model_list.append(score_base_model)
                runtime_list.append(np.mean(runtime_list_cv))
            else:

                dataset_dict = dataset_dict_list[i]
                
                ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))      


                if config['preprocessing']['XGBoostEncoding']:
                    feature_types = []
                    for feature_index in range(X_train.shape[1]):
                        if feature_index in dataset_dict['categorical_feature_indices']:
                            feature_types.append('c')
                        else:
                            feature_types.append('q')

                    base_model = model_class(enable_categorical=True, feature_types=feature_types)                      
                else:
                    base_model = model_class()                

                
                base_model.set_params(**parameter_setting)

                start = timeit.default_timer()
                if config['GRANDE']['class_weights']:
                    base_model.fit(enforce_numpy(X_train), 
                                   enforce_numpy(y_train),
                                   sample_weight=calculate_sample_weights(enforce_numpy(y_train)),
                                   eval_set=[(enforce_numpy(X_valid), enforce_numpy(y_valid))], 
                                   sample_weight_eval_set=[calculate_sample_weights(enforce_numpy(y_valid))], 
                                   #eval_metric='auc' if config['GRANDE']['objective'] == 'classification' else 'mae',
                                   verbose=False)
                else:
                    base_model.fit(enforce_numpy(X_train), 
                                   enforce_numpy(y_train),
                                   eval_set=[(enforce_numpy(X_valid), enforce_numpy(y_valid))], 
                                   #eval_metric='auc' if config['GRANDE']['objective'] == 'classification' else 'mae',
                                   verbose=False)                    

                end = timeit.default_timer()  
                runtime = end - start                

                base_model_pred = base_model.predict(enforce_numpy(X_test))
                base_model_pred = np.nan_to_num(base_model_pred)
                if config['GRANDE']['objective'] == 'classification':
                    base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                    base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                    if metric not in ['f1', 'roc_auc', 'accuracy']:
                        score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))
                    else:
                        if metric == 'f1':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                        if metric == 'accuracy':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                                
                        elif metric == 'roc_auc':
                            try:
                                if number_of_classes > 2:                            
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                else:
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                            except ValueError:
                                score_base_model = 0.5
                else:
                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                score_base_model_list.append(score_base_model)
                runtime_list.append(runtime)                
                
        
        if config['computation']['report_test_performance']:
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_test_raw_list = []
                runtime_test_raw_list = []                       

                for i in range(config['computation']['cv_num_eval']):   

                    dataset_dict = dataset_dict_list[i][0]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))              

                    if config['preprocessing']['XGBoostEncoding']:
                        feature_types = []
                        for feature_index in range(X_train.shape[1]):
                            if feature_index in dataset_dict['categorical_feature_indices']:
                                feature_types.append('c')
                            else:
                                feature_types.append('q')

                        base_model = model_class(enable_categorical=True, feature_types=feature_types)                      
                    else:
                        base_model = model_class()                      
                    
                    base_model.set_params(**parameter_setting)

                    start = timeit.default_timer()
                    if config['GRANDE']['class_weights']:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train),
                                       sample_weight=calculate_sample_weights(enforce_numpy(y_train)),
                                       eval_set=[(enforce_numpy(X_valid), enforce_numpy(y_valid))], 
                                       sample_weight_eval_set=[calculate_sample_weights(enforce_numpy(y_valid))],  
                                       #eval_metric='auc' if config['GRANDE']['objective'] == 'classification' else 'mae',
                                       verbose=False)
                    else:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train),
                                       eval_set=[(enforce_numpy(X_valid), enforce_numpy(y_valid))], 
                                       #eval_metric='auc' if config['GRANDE']['objective'] == 'classification' else 'mae',
                                       verbose=False)                    
                    
                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)
                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_test_raw = 0.5

                    else:
                        score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_test_raw_list.append(score_base_model_test_raw)
                    runtime_test_raw_list.append(runtime)
      
            else:
                score_base_model_test_raw_list = score_base_model_list
        else:
            score_base_model_test_raw_list = score_base_model_list                
        
        return {
            'cv_score': np.mean(score_base_model_list),
            'test_score': np.mean(score_base_model_test_raw_list),
            'runtime': np.mean(runtime_list),
            'parameter_setting': parameter_setting,
        }

    if config['computation']['verbose'] > 0:
        print('Number of Trials XGB: ' + str(config['computation']['search_iterations']))
        
    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
        hpo_mode = "max" if metric in ['r2', 'accuracy', 'f1', 'roc_auc'] else "min"

        initial_params_update = [{
                'learning_rate': 0.1, 
                'max_depth': 6, 
            
                'reg_alpha': 0, 
                'reg_lambda': 1, 
        }]  

        initial_params = [{key: value.sample() for key, value in parameter_dict.items()}]
        initial_params[0].update(initial_params_update[0])
                
        
        optuna_search = OptunaSearch(metric="cv_score", mode=hpo_mode, points_to_evaluate=initial_params, seed=config['computation']['random_seed'])

        config_ray = {
                      'parameters': parameter_dict,
                      'dataset_dict_list': dataset_dict_list, 
                      'config': config,
                     }


        print_intermediate_tables = True if config['computation']['verbose'] > 2 else False 
        progress_reporter = JupyterNotebookReporter(overwrite=False,metric_columns=['cv_score','test_score','runtime'],max_progress_rows=config['computation']['search_iterations'],print_intermediate_tables=False,max_column_length=50) if config['computation']['verbose'] > 0 else JupyterNotebookReporter(overwrite=False, max_progress_rows=0, metric_columns=[], parameter_columns=[])

        analysis = tune.run(
                            evaluate_parameter_setting_XGB,
                            config=config_ray,
                            search_alg=optuna_search,
                            resources_per_trial={"cpu": config['computation']['n_jobs'], 'gpu': config['computation']['num_gpu']},
                            num_samples=config['computation']['search_iterations'],
                            #log_to_file=False,
                            local_dir='./results_ray/' + timestr + '/',
                            #callbacks=[],
                            raise_on_failed_trial=False,        
                            verbose=config['computation']['verbose'],
                            progress_reporter=progress_reporter, #,parameter_columns=['cv_score','test_score','runtime']        
                            #progress_reporter=CustomProgressReporter(metric_columns=['cv_score','test_score','runtime']),
                            #keep_checkpoints_num=1,
                            #checkpoint_score_attr='cv_score'
                           )    

    grid_search_results = [tuple(trial.last_result.values()) for trial in analysis.trials]          
           
    replace_value = -np.inf if greater_better else np.inf

    replace_dict = {}
    for some_dict in [result[3] for result in grid_search_results]:
        if isinstance(some_dict, dict):
            for key in some_dict.keys():
                replace_dict[key] = None 
            break

    grid_search_results = [[np.nan_to_num(pd.to_numeric(result[0], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[1], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[2], errors='coerce'), nan=replace_value), 
                            result[3] if isinstance(result[3], dict) else replace_dict] for result in grid_search_results]
    
    cv_scores_base_model = np.nan_to_num([result[0] for result in grid_search_results], nan=replace_value)
    test_scores_base_model = np.nan_to_num([result[1] for result in grid_search_results], nan=replace_value)
    runtime_base_model = np.nan_to_num([result[2] for result in grid_search_results], nan=np.inf)
    parameter_settings = [result[3] for result in grid_search_results]


    grid_search_results_sorted = sorted(grid_search_results, key=lambda tup: tup[0], reverse=greater_better) 

    best_score_cv = grid_search_results_sorted[0][0]
    best_score_test_raw = grid_search_results_sorted[0][1]
    best_model_runtime_cv = grid_search_results_sorted[0][2]
    best_params = grid_search_results_sorted[0][3]
   
    if config['GRANDE']['objective'] == 'classification':
        model_class = XGBClassifier
        model_identifier = 'XGB_class'
    else:
        model_class = XGBRegressor      
        model_identifier = 'XGB_reg'
            
    if config['computation']['report_test_performance']:
        with_valid = True
        encoded = False if config['preprocessing']['XGBoostEncoding'] else True
                
        runtime_list = []
        base_model_list = []           
        dataset_dict_list = ray.get(dataset_dict_list)
        for i in range(config['computation']['cv_num_eval']):
            try:
                dataset_dict = dataset_dict_list[i][0]
            except:
                dataset_dict = dataset_dict_list[i]
                
            ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

            number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

            if config['preprocessing']['XGBoostEncoding']:
                feature_types = []
                for feature_index in range(X_train.shape[1]):
                    if feature_index in dataset_dict['categorical_feature_indices']:
                        feature_types.append('c')
                    else:
                        feature_types.append('q')

                base_model = model_class(enable_categorical=True, feature_types=feature_types)                      
            else:
                base_model = model_class()             

            base_model.set_params(**best_params)

            start = timeit.default_timer()
            
            if config['GRANDE']['class_weights']:
                base_model.fit(enforce_numpy(X_train), 
                               enforce_numpy(y_train),
                               sample_weight=calculate_sample_weights(enforce_numpy(y_train)),
                               eval_set=[(enforce_numpy(X_valid), enforce_numpy(y_valid))], 
                               sample_weight_eval_set=[calculate_sample_weights(enforce_numpy(y_valid))],
                               #eval_metric='auc' if config['GRANDE']['objective'] == 'classification' else 'mae',
                               verbose=False)    
            else:
                base_model.fit(enforce_numpy(X_train), 
                               enforce_numpy(y_train),
                               eval_set=[(enforce_numpy(X_valid), enforce_numpy(y_valid))], 
                               #eval_metric='auc' if config['GRANDE']['objective'] == 'classification' else 'mae',
                               verbose=False)                    
            end = timeit.default_timer()  
            runtime = end - start
            
            runtime_list.append(runtime)
            base_model_list.append(base_model)
        
    else:
        runtime_list = [best_model_runtime_cv for _ in range(config['computation']['cv_num_eval'])]
        base_model_list = [None for _ in range(config['computation']['cv_num_eval'])]
    
    if config['computation']['verbose'] > 0:
        print('Best Params XGB')
        display(best_params)        
        print('Best CV-Score XGB:', best_score_cv)

    hpo_path_by_dataset =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/' + identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/').mkdir(parents=True, exist_ok=True)    
        
    if not os.path.isfile(hpo_path_by_dataset):
        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()        
    
    hpo_path_by_dataset_with_timestamp =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/' + model_identifier + '___' +  identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/').mkdir(parents=True, exist_ok=True)    
    
    if not os.path.isfile(hpo_path_by_dataset_with_timestamp):
        file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()      
    
    return (best_score_cv, runtime_list, base_model_list), cv_scores_base_model, test_scores_base_model

    
    
def hpo_CatBoost(identifier, dataset_dict_list, parameter_dict, config, metric='f1', greater_better=True, timestr = 'placeholder'):

    def evaluate_parameter_setting_CatBoost(trial_config):   
              
        parameter_setting = trial_config['parameters']
        dataset_dict_list = ray.get(trial_config['dataset_dict_list'])
        config = trial_config['config']
        
        with_valid = True
        encoded = False if config['preprocessing']['CatBoostEncoding'] else True        
        
        score_base_model_list = []
        runtime_list = []     
        
        if config['GRANDE']['objective'] == 'classification':
             model_class = CatBoostClassifier
        else:
             model_class = CatBoostRegressor  
                            
        for i in range(config['computation']['cv_num_eval']):   
            
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_cv_list = []
                runtime_list_cv = []                  
                for j in range(config['computation']['cv_num_hpo']):

                    dataset_dict = dataset_dict_list[i][j]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded, cv=True)
                    
                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

                    cat_features = dataset_dict['categorical_feature_indices'] if config['preprocessing']['CatBoostEncoding'] else None

                    if config['GRANDE']['objective'] == 'classification':
                        if config['GRANDE']['class_weights']:
                            train_data = Pool(
                                    data=enforce_numpy(X_train),
                                    label=enforce_numpy(y_train),
                                    cat_features=cat_features,
                                    weight=calculate_sample_weights(enforce_numpy(y_train))
                                )

                            eval_data = Pool(
                                    data=enforce_numpy(X_valid),
                                    label=enforce_numpy(y_valid),
                                    cat_features=cat_features,
                                    weight=calculate_sample_weights(enforce_numpy(y_valid))
                                ) 
                        else:
                            train_data = Pool(
                                    data=enforce_numpy(X_train),
                                    label=enforce_numpy(y_train),
                                    cat_features=cat_features,
                                )

                            eval_data = Pool(
                                    data=enforce_numpy(X_valid),
                                    label=enforce_numpy(y_valid),
                                    cat_features=cat_features,
                                )                                 

                    else:
                        train_data = Pool(
                                data=enforce_numpy(X_train),
                                label=enforce_numpy(y_train),
                                cat_features=cat_features,
                            )

                        eval_data = Pool(
                                data=enforce_numpy(X_valid),
                                label=enforce_numpy(y_valid),
                                cat_features=cat_features,
                            )                      
                    
                    base_model = model_class()
                    
                    
                    base_model.set_params(**parameter_setting)

                    start = timeit.default_timer()
                    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                        base_model.fit(X=train_data, eval_set=eval_data)

                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)
                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_cv = 0.5
                    
                    else:
                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_cv_list.append(score_base_model_cv)
                    runtime_list_cv.append(runtime)

                score_base_model = np.mean(score_base_model_cv_list)

                score_base_model_list.append(score_base_model)
                runtime_list.append(np.mean(runtime_list_cv))
            else:

                dataset_dict = dataset_dict_list[i]
                
                ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))
                cat_features = dataset_dict['categorical_feature_indices'] if config['preprocessing']['CatBoostEncoding'] else None

                if config['GRANDE']['objective'] == 'classification':
                    if config['GRANDE']['class_weights']:
                        train_data = Pool(
                                data=enforce_numpy(X_train),
                                label=enforce_numpy(y_train),
                                cat_features=cat_features,
                                weight=calculate_sample_weights(enforce_numpy(y_train))
                            )

                        eval_data = Pool(
                                data=enforce_numpy(X_valid),
                                label=enforce_numpy(y_valid),
                                cat_features=cat_features,
                                weight=calculate_sample_weights(enforce_numpy(y_valid))
                            ) 
                    else:
                        train_data = Pool(
                                data=enforce_numpy(X_train),
                                label=enforce_numpy(y_train),
                                cat_features=cat_features,
                            )

                        eval_data = Pool(
                                data=enforce_numpy(X_valid),
                                label=enforce_numpy(y_valid),
                                cat_features=cat_features,
                            )                                 

                else:
                    train_data = Pool(
                            data=enforce_numpy(X_train),
                            label=enforce_numpy(y_train),
                            cat_features=cat_features,
                        )

                    eval_data = Pool(
                            data=enforce_numpy(X_valid),
                            label=enforce_numpy(y_valid),
                            cat_features=cat_features,
                        )                     

                base_model = model_class()

                base_model.set_params(**parameter_setting)

                start = timeit.default_timer()
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
                        base_model.fit(X=train_data, eval_set=eval_data)

                end = timeit.default_timer()  
                runtime = end - start                     

                base_model_pred = base_model.predict(enforce_numpy(X_test))
                base_model_pred = np.nan_to_num(base_model_pred)
                if config['GRANDE']['objective'] == 'classification':
                    base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                    base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                    if metric not in ['f1', 'roc_auc', 'accuracy']:
                        score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))
                    else:
                        if metric == 'f1':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                        if metric == 'accuracy':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                                
                        elif metric == 'roc_auc':
                            try:
                                if number_of_classes > 2:                            
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                else:
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                            except ValueError:
                                score_base_model = 0.5
                else:
                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                score_base_model_list.append(score_base_model)
                runtime_list.append(runtime)                
                
        
        if config['computation']['report_test_performance']:
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_test_raw_list = []
                runtime_test_raw_list = []                       

                for i in range(config['computation']['cv_num_eval']):   

                    dataset_dict = dataset_dict_list[i][0]

                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)
                    
                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))
                    cat_features = dataset_dict['categorical_feature_indices'] if config['preprocessing']['CatBoostEncoding'] else None

                    if config['GRANDE']['objective'] == 'classification':
                        if config['GRANDE']['class_weights']:
                            train_data = Pool(
                                    data=enforce_numpy(X_train),
                                    label=enforce_numpy(y_train),
                                    cat_features=cat_features,
                                    weight=calculate_sample_weights(enforce_numpy(y_train))
                                )

                            eval_data = Pool(
                                    data=enforce_numpy(X_valid),
                                    label=enforce_numpy(y_valid),
                                    cat_features=cat_features,
                                    weight=calculate_sample_weights(enforce_numpy(y_valid))
                                ) 
                        else:
                            train_data = Pool(
                                    data=enforce_numpy(X_train),
                                    label=enforce_numpy(y_train),
                                    cat_features=cat_features,
                                )

                            eval_data = Pool(
                                    data=enforce_numpy(X_valid),
                                    label=enforce_numpy(y_valid),
                                    cat_features=cat_features,
                                )                                 

                    else:
                        train_data = Pool(
                                data=enforce_numpy(X_train),
                                label=enforce_numpy(y_train),
                                cat_features=cat_features,
                            )

                        eval_data = Pool(
                                data=enforce_numpy(X_valid),
                                label=enforce_numpy(y_valid),
                                cat_features=cat_features,
                            )                      
                    
                    base_model = model_class()
                    
                    
                    base_model.set_params(**parameter_setting)

                    start = timeit.default_timer()
                    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                        with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
                            base_model.fit(X=train_data, eval_set=eval_data)

                    end = timeit.default_timer()  
                    runtime = end - start               

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)
                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_test_raw = 0.5

                    else:
                        score_base_model_test_raw = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_test_raw_list.append(score_base_model_test_raw)
                    runtime_test_raw_list.append(runtime)
      
            else:
                score_base_model_test_raw_list = score_base_model_list
        else:
            score_base_model_test_raw_list = score_base_model_list                
        
        return {
            'cv_score': np.mean(score_base_model_list),
            'test_score': np.mean(score_base_model_test_raw_list),
            'runtime': np.mean(runtime_list),
            'parameter_setting': parameter_setting,
        }


    if config['computation']['verbose'] > 0:
        print('Number of Trials CatBoost: ' + str(config['computation']['search_iterations']))
    
    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
        hpo_mode = "max" if metric in ['r2', 'accuracy', 'f1', 'roc_auc'] else "min"

        initial_params_update = [{
                'learning_rate': 0.1,#TabSurvey
                'max_depth': 6,#TabSurvey
                'l2_leaf_reg': 3,#TabSurvey
        }]   

        initial_params = [{key: value.sample() for key, value in parameter_dict.items()}]
        initial_params[0].update(initial_params_update[0])
        
        optuna_search = OptunaSearch(metric="cv_score", mode=hpo_mode, points_to_evaluate=initial_params, seed=config['computation']['random_seed'])

        config_ray = {
                      'parameters': parameter_dict,
                      'dataset_dict_list': dataset_dict_list, 
                      'config': config,
                     }

        print_intermediate_tables = True if config['computation']['verbose'] > 2 else False 
        progress_reporter = JupyterNotebookReporter(overwrite=False,metric_columns=['cv_score','test_score','runtime'],max_progress_rows=config['computation']['search_iterations'],print_intermediate_tables=False,max_column_length=50) if config['computation']['verbose'] > 0 else JupyterNotebookReporter(overwrite=False, max_progress_rows=0, metric_columns=[], parameter_columns=[])

        analysis = tune.run(
                            evaluate_parameter_setting_CatBoost,
                            config=config_ray,
                            search_alg=optuna_search,
                            resources_per_trial={"cpu": config['computation']['n_jobs'], 'gpu': config['computation']['num_gpu']},
                            num_samples=config['computation']['search_iterations'],
                            #log_to_file=False,
                            local_dir='./results_ray/' + timestr + '/',
                            #callbacks=[],
                            raise_on_failed_trial=False,        
                            verbose=config['computation']['verbose'],
                            progress_reporter=progress_reporter, #,parameter_columns=['cv_score','test_score','runtime']        
                            #progress_reporter=CustomProgressReporter(metric_columns=['cv_score','test_score','runtime']),
                            #keep_checkpoints_num=1,
                            #checkpoint_score_attr='cv_score'
                           )    

    grid_search_results = [tuple(trial.last_result.values()) for trial in analysis.trials]          
           
    replace_value = -np.inf if greater_better else np.inf

    replace_dict = {}
    for some_dict in [result[3] for result in grid_search_results]:
        if isinstance(some_dict, dict):
            for key in some_dict.keys():
                replace_dict[key] = None 
            break

    grid_search_results = [[np.nan_to_num(pd.to_numeric(result[0], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[1], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[2], errors='coerce'), nan=replace_value), 
                            result[3] if isinstance(result[3], dict) else replace_dict] for result in grid_search_results]
    
    cv_scores_base_model = np.nan_to_num([result[0] for result in grid_search_results], nan=replace_value)
    test_scores_base_model = np.nan_to_num([result[1] for result in grid_search_results], nan=replace_value)
    runtime_base_model = np.nan_to_num([result[2] for result in grid_search_results], nan=np.inf)
    parameter_settings = [result[3] for result in grid_search_results]


    grid_search_results_sorted = sorted(grid_search_results, key=lambda tup: tup[0], reverse=greater_better) 

    best_score_cv = grid_search_results_sorted[0][0]
    best_score_test_raw = grid_search_results_sorted[0][1]
    best_model_runtime_cv = grid_search_results_sorted[0][2]
    best_params = grid_search_results_sorted[0][3]
   
    if config['GRANDE']['objective'] == 'classification':
        model_class = CatBoostClassifier
        model_identifier = 'CatBoost_class'
    else:
        model_class = CatBoostRegressor      
        model_identifier = 'CatBoost_reg'
            
    if config['computation']['report_test_performance']:
        with_valid = True
        encoded = False if config['preprocessing']['CatBoostEncoding'] else True
        
        runtime_list = []
        base_model_list = []           
        dataset_dict_list = ray.get(dataset_dict_list)
        for i in range(config['computation']['cv_num_eval']):
            try:
                dataset_dict = dataset_dict_list[i][0]
            except:
                dataset_dict = dataset_dict_list[i]
                
            ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

            number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

            cat_features = dataset_dict['categorical_feature_indices'] if config['preprocessing']['CatBoostEncoding'] else None

            if config['GRANDE']['objective'] == 'classification':
                if config['GRANDE']['class_weights']:
                    train_data = Pool(
                            data=enforce_numpy(X_train),
                            label=enforce_numpy(y_train),
                            cat_features=cat_features,
                            weight=calculate_sample_weights(enforce_numpy(y_train))
                        )

                    eval_data = Pool(
                            data=enforce_numpy(X_valid),
                            label=enforce_numpy(y_valid),
                            cat_features=cat_features,
                            weight=calculate_sample_weights(enforce_numpy(y_valid))
                        ) 
                else:
                    train_data = Pool(
                            data=enforce_numpy(X_train),
                            label=enforce_numpy(y_train),
                            cat_features=cat_features,
                        )

                    eval_data = Pool(
                            data=enforce_numpy(X_valid),
                            label=enforce_numpy(y_valid),
                            cat_features=cat_features,
                        )                                 

            else:
                train_data = Pool(
                        data=enforce_numpy(X_train),
                        label=enforce_numpy(y_train),
                        cat_features=cat_features,
                    )

                eval_data = Pool(
                        data=enforce_numpy(X_valid),
                        label=enforce_numpy(y_valid),
                        cat_features=cat_features,
                    )                      

            base_model = model_class()


            base_model.set_params(**best_params)

            start = timeit.default_timer()
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
                    base_model.fit(X=train_data, eval_set=eval_data)

            end = timeit.default_timer()  
            runtime = end - start       

            runtime_list.append(runtime)
            base_model_list.append(base_model)
        
    else:
        runtime_list = [best_model_runtime_cv for _ in range(config['computation']['cv_num_eval'])]
        base_model_list = [None for _ in range(config['computation']['cv_num_eval'])]
    
    if config['computation']['verbose'] > 0:
        print('Best Params CatBoost')
        display(best_params)        
        print('Best CV-Score CatBoost:', best_score_cv)

    hpo_path_by_dataset =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/' + identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/').mkdir(parents=True, exist_ok=True)    
        
    if not os.path.isfile(hpo_path_by_dataset):
        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()        
    
    hpo_path_by_dataset_with_timestamp =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/' + model_identifier + '___' +  identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/').mkdir(parents=True, exist_ok=True)    
    
    if not os.path.isfile(hpo_path_by_dataset_with_timestamp):
        file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()        
    
    return (best_score_cv, runtime_list, base_model_list), cv_scores_base_model, test_scores_base_model
  

def hpo_NODE(identifier, dataset_dict_list, parameter_dict, config, metric='f1', greater_better=True, timestr = 'placeholder'):

    def evaluate_parameter_setting_NODE(trial_config):

        parameter_setting = trial_config['parameters']
        dataset_dict_list = ray.get(trial_config['dataset_dict_list'])
        config = trial_config['config'] 
        
        with_valid = True
        encoded = True         

        score_base_model_list = []
        runtime_list = []     

        for i in range(config['computation']['cv_num_eval']):   

            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_cv_list = []
                runtime_list_cv = []                  
                for j in range(config['computation']['cv_num_hpo']):

                    dataset_dict = dataset_dict_list[i][j]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded, cv=True)

                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

                    start = timeit.default_timer()

                    args = {'num_features': dataset_dict['X_train'].shape[1],
                                           'objective': 'binary' if config['GRANDE']['objective'] == 'classification' and number_of_classes==2 else config_training['GRANDE']['objective'],#config_training['GRANDE']['objective'],
                                           'num_classes': 1 if config['GRANDE']['objective'] == 'classification' and number_of_classes==2 else number_of_classes,
                                           'use_gpu': config['computation']['use_gpu'],
                                           'data_parallel': False,
                                           'early_stopping_rounds': 20,
                                           'logging_period': 100,
                                           'batch_size': config['GRANDE']['batch_size'],
                                           'val_batch_size': config['GRANDE']['batch_size'],
                                           'epochs': 1000}                                 
                    
                    args = MyDictObject(**args)

                    torch.manual_seed(config['computation']['random_seed'])
                    random.seed(config['computation']['random_seed'])
                    np.random.seed(config['computation']['random_seed'])    
                    base_model = NODE(parameter_setting, args)

                    base_model.fit(X=enforce_numpy(X_train, dtype='float'),
                                          y=enforce_numpy(y_train, dtype='float'),
                                          X_val=enforce_numpy(X_valid, dtype='float'),
                                          y_val=enforce_numpy(y_valid, dtype='float'))                    
                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)

                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(pd.DataFrame(X_test))[:,1]
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)
                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_cv = 0.5

                    else:
                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_cv_list.append(score_base_model_cv)
                    runtime_list_cv.append(runtime)

                score_base_model = np.mean(score_base_model_cv_list)

                score_base_model_list.append(score_base_model)
                runtime_list.append(np.mean(runtime_list_cv))
                free_memory()
            else:

                dataset_dict = dataset_dict_list[i]
                
                ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))      

                start = timeit.default_timer()

                args = {'num_features': X_train.shape[1],
                       'objective': config['GRANDE']['objective'],
                       'num_classes': number_of_classes,
                       'use_gpu': config['computation']['use_gpu'],
                       'data_parallel': False,
                       'early_stopping_rounds': 20,
                       'logging_period': 100,
                        'batch_size': config['GRANDE']['batch_size'],
                       'val_batch_size': config['GRANDE']['batch_size'],
                       'epochs': 1000}    

                args = MyDictObject(**args)

                torch.manual_seed(config['computation']['random_seed'])
                random.seed(config['computation']['random_seed'])
                np.random.seed(config['computation']['random_seed'])   
                base_model = NODE(parameter_setting, args)

                base_model.fit(X=enforce_numpy(X_train, dtype='float'),
                                      y=enforce_numpy(y_train, dtype='float'),
                                      X_val=enforce_numpy(X_valid, dtype='float'),
                                      y_val=enforce_numpy(y_valid, dtype='float'))                 

                end = timeit.default_timer()  
                runtime = end - start                

                base_model_pred = base_model.predict(enforce_numpy(X_test))
                base_model_pred = np.nan_to_num(base_model_pred)
                if config['GRANDE']['objective'] == 'classification':
                    base_model_pred_proba = base_model.predict_proba(pd.DataFrame(X_test))[:,1]
                    base_model_pred_proba = np.nan_to_num(base_model_pred_proba)

                    if metric not in ['f1', 'roc_auc', 'accuracy']:
                        score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))
                    else:
                        if metric == 'f1':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                        if metric == 'accuracy':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                                
                        elif metric == 'roc_auc':
                            try:
                                if number_of_classes > 2:                            
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                else:
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                            except ValueError:
                                score_base_model = 0.5
                else:
                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                score_base_model_list.append(score_base_model)
                runtime_list.append(runtime)      
                free_memory()


        if config['computation']['report_test_performance']:
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_test_list = []
                runtime_test_list = []                       

                for i in range(config['computation']['cv_num_eval']):   

                    dataset_dict = dataset_dict_list[i][0]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))              

                    start = timeit.default_timer()

                    args = {'num_features': dataset_dict['X_train'].shape[1],
                                           'objective': 'binary' if config['GRANDE']['objective'] == 'classification' and number_of_classes==2 else config_training['GRANDE']['objective'],#config_training['GRANDE']['objective'],
                                           'num_classes': 1 if config['GRANDE']['objective'] == 'classification' and number_of_classes==2 else number_of_classes,
                                           'use_gpu': config['computation']['use_gpu'],
                                           'data_parallel': False,
                                           'early_stopping_rounds': 20,
                                           'logging_period': 100,
                                           'batch_size': config['GRANDE']['batch_size'],
                                           'val_batch_size': config['GRANDE']['batch_size'],
                                           'epochs': 1000}    

                    args = MyDictObject(**args)

                    torch.manual_seed(config['computation']['random_seed'])
                    random.seed(config['computation']['random_seed'])
                    np.random.seed(config['computation']['random_seed'])   

                    base_model = NODE(parameter_setting, args)

                    base_model.fit(X=enforce_numpy(X_train, dtype='float'),
                                          y=enforce_numpy(y_train, dtype='float'),
                                          X_val=enforce_numpy(X_valid, dtype='float'),
                                          y_val=enforce_numpy(y_valid, dtype='float')) 

                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)

                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(pd.DataFrame(X_test))[:,1]
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)

                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_test = 0.5

                    else:
                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_test_list.append(score_base_model_test)
                    runtime_test_list.append(runtime)
                    free_memory()

            else:
                score_base_model_test_list = score_base_model_list
        else:
            score_base_model_test_list = score_base_model_list                

        return {
            'cv_score': np.mean(score_base_model_list),
            'test_score': np.mean(score_base_model_test_list),
            'runtime': np.mean(runtime_list),
            'parameter_setting': parameter_setting,
        }

   
    if config['computation']['verbose'] > 0:
        print('Number of Trials NODE: ' + str(config['computation']['search_iterations']))
    
    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
        hpo_mode = "max" if metric in ['r2', 'accuracy', 'f1', 'roc_auc'] else "min"
        optuna_search = OptunaSearch(metric="cv_score", mode=hpo_mode, points_to_evaluate=[], seed=config['computation']['random_seed'])

        config_ray = {
                      'parameters': parameter_dict,
                      'dataset_dict_list': dataset_dict_list, 
                      'config': config,
                     }

        print_intermediate_tables = True if config['computation']['verbose'] > 2 else False 
        progress_reporter = JupyterNotebookReporter(overwrite=False,metric_columns=['cv_score','test_score','runtime'],max_progress_rows=config['computation']['search_iterations'],print_intermediate_tables=False,max_column_length=50) if config['computation']['verbose'] > 0 else JupyterNotebookReporter(overwrite=False, max_progress_rows=0, metric_columns=[], parameter_columns=[])

        parameter_size_list = [len(value) for value in parameter_dict.values()]
        num_samples = reduce(lambda x, y: x * y, parameter_size_list)        
        
        num_samples = min(config['computation']['search_iterations'], num_samples)
        
        analysis = tune.run(
                            evaluate_parameter_setting_NODE,
                            config=config_ray,
                            search_alg=optuna_search,
                            resources_per_trial={"cpu": config['computation']['n_jobs'], 'gpu': config['computation']['num_gpu']},
                            num_samples=num_samples,
                            #log_to_file=False,
                            local_dir='./results_ray/' + timestr + '/',
                            #callbacks=[],
                            raise_on_failed_trial=False,        
                            verbose=config['computation']['verbose'],
                            progress_reporter=progress_reporter, #,parameter_columns=['cv_score','test_score','runtime']        
                            #progress_reporter=CustomProgressReporter(metric_columns=['cv_score','test_score','runtime']),
                            #keep_checkpoints_num=1,
                            #checkpoint_score_attr='cv_score'
                           )    

    grid_search_results = [tuple(trial.last_result.values()) for trial in analysis.trials]          
           
    replace_value = -np.inf if greater_better else np.inf

    replace_dict = {}
    for some_dict in [result[3] for result in grid_search_results]:
        if isinstance(some_dict, dict):
            for key in some_dict.keys():
                replace_dict[key] = None 
            break

    grid_search_results = [[np.nan_to_num(pd.to_numeric(result[0], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[1], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[2], errors='coerce'), nan=replace_value), 
                            result[3] if isinstance(result[3], dict) else replace_dict] for result in grid_search_results]
    
    cv_scores_base_model = np.nan_to_num([result[0] for result in grid_search_results], nan=replace_value)
    test_scores_base_model = np.nan_to_num([result[1] for result in grid_search_results], nan=replace_value)
    runtime_base_model = np.nan_to_num([result[2] for result in grid_search_results], nan=np.inf)
    parameter_settings = [result[3] for result in grid_search_results]

    grid_search_results_sorted = sorted(grid_search_results, key=lambda tup: tup[0], reverse=greater_better) 

    best_score_cv = grid_search_results_sorted[0][0]
    best_score_test = grid_search_results_sorted[0][1]
    best_model_runtime_cv = grid_search_results_sorted[0][2]
    best_params = grid_search_results_sorted[0][3]
       
    model_identifier = 'NODE'
            
    if config['computation']['report_test_performance']:
        with_valid = True
        encoded = True                       
        
        runtime_list = []
        base_model_list = []           
        dataset_dict_list = ray.get(dataset_dict_list)
        for i in range(config['computation']['cv_num_eval']):
            try:
                dataset_dict = dataset_dict_list[i][0]
            except:
                dataset_dict = dataset_dict_list[i]
                
            ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

            number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

            start = timeit.default_timer()
            args = {'num_features': X_train.shape[1],
                   'objective': config['GRANDE']['objective'],
                   'num_classes': number_of_classes,
                   'use_gpu': config['computation']['use_gpu'],
                   'data_parallel': False,
                   'early_stopping_rounds': 20,
                   'logging_period': 100,
                   'batch_size': config['GRANDE']['batch_size'],
                   'val_batch_size': config['GRANDE']['batch_size'],
                   'epochs': 1000}       
            
            args = MyDictObject(**args)

            torch.manual_seed(config['computation']['random_seed'])
            random.seed(config['computation']['random_seed'])
            np.random.seed(config['computation']['random_seed'])               
            base_model = NODE(best_params, args)

            base_model.fit(X=enforce_numpy(X_train, dtype='float'),
                                  y=enforce_numpy(y_train, dtype='float'),
                                  X_val=enforce_numpy(X_valid, dtype='float'),
                                  y_val=enforce_numpy(y_valid, dtype='float'))     
            end = timeit.default_timer()  
            runtime = end - start
            
            runtime_list.append(runtime)
            base_model_list.append(base_model)
        
    else:
        runtime_list = [best_model_runtime_cv for _ in range(config['computation']['cv_num_eval'])]
        base_model_list = [None for _ in range(config['computation']['cv_num_eval'])]
    
    if config['computation']['verbose'] > 0:
        print('Best Params NODE')
        display(best_params)        
        print('Best CV-Score NODE:', best_score_cv)

    hpo_path_by_dataset =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/' + identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/').mkdir(parents=True, exist_ok=True)    
        
    if not os.path.isfile(hpo_path_by_dataset):
        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()        
    
    hpo_path_by_dataset_with_timestamp =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/' + model_identifier + '___' +  identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/').mkdir(parents=True, exist_ok=True)    
    
    if not os.path.isfile(hpo_path_by_dataset_with_timestamp):
        file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()           
    
    return (best_score_cv, runtime_list, base_model_list), cv_scores_base_model, test_scores_base_model

def hpo_CART(identifier, dataset_dict_list, parameter_dict, config, metric='f1', greater_better=True, timestr = 'placeholder'):
    
    def evaluate_parameter_setting_CART(trial_config):   
        
        with_valid = False
        encoded = True
        
        parameter_setting = trial_config['parameters']
        dataset_dict_list = ray.get(trial_config['dataset_dict_list'])
        config = trial_config['config']
        
        score_base_model_list = []
        runtime_list = []     
        
        if config['GRANDE']['objective'] == 'classification':
             model_class = DecisionTreeClassifier
        else:
             model_class = DecisionTreeRegressor  
                            
        for i in range(config['computation']['cv_num_eval']):   
            
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_cv_list = []
                runtime_list_cv = []                  
                for j in range(config['computation']['cv_num_hpo']):

                    dataset_dict = dataset_dict_list[i][j]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded, cv=True)

                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

                    base_model = model_class()
                    base_model.set_params(**parameter_setting)

                    start = timeit.default_timer()
                    if config['GRANDE']['class_weights']:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train),
                                       sample_weight=calculate_sample_weights(enforce_numpy(y_train)))
                    else:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train))                        

                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)

                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)

                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_cv = 0.5
                    
                    else:
                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_cv_list.append(score_base_model_cv)
                    runtime_list_cv.append(runtime)

                score_base_model = np.mean(score_base_model_cv_list)

                score_base_model_list.append(score_base_model)
                runtime_list.append(np.mean(runtime_list_cv))
            else:

                dataset_dict = dataset_dict_list[i]
                
                ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))      

                base_model = model_class()
                base_model.set_params(**parameter_setting)

                start = timeit.default_timer()
                if config['GRANDE']['class_weights']:
                    base_model.fit(enforce_numpy(X_train), 
                                   enforce_numpy(y_train),
                                   sample_weight=calculate_sample_weights(enforce_numpy(y_train)))
                else:
                    base_model.fit(enforce_numpy(X_train), 
                                   enforce_numpy(y_train))                    

                end = timeit.default_timer()  
                runtime = end - start                

                base_model_pred = base_model.predict(enforce_numpy(X_test))
                base_model_pred = np.nan_to_num(base_model_pred)

                if config['GRANDE']['objective'] == 'classification':
                    base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                    base_model_pred_proba = np.nan_to_num(base_model_pred_proba)

                    if metric not in ['f1', 'roc_auc', 'accuracy']:
                        score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))
                    else:
                        if metric == 'f1':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                        if metric == 'accuracy':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                                
                        elif metric == 'roc_auc':
                            try:
                                if number_of_classes > 2:                            
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                else:
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                            except ValueError:
                                score_base_model = 0.5
                else:
                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                score_base_model_list.append(score_base_model)
                runtime_list.append(runtime)                
                
        
        if config['computation']['report_test_performance']:
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_test_list = []
                runtime_test_list = []                       

                for i in range(config['computation']['cv_num_eval']):   

                    dataset_dict = dataset_dict_list[i][0]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))              

                    base_model = model_class()
                    base_model.set_params(**parameter_setting)

                    start = timeit.default_timer()
                    if config['GRANDE']['class_weights']:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train),
                                       sample_weight=calculate_sample_weights(enforce_numpy(y_train)))
                    else:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train))                        

                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)

                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)

                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_test = 0.5

                    else:
                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_test_list.append(score_base_model_test)
                    runtime_test_list.append(runtime)
      
            else:
                score_base_model_test_list = score_base_model_list
        else:
            score_base_model_test_list = score_base_model_list                
        
        return {
            'cv_score': np.mean(score_base_model_list),
            'test_score': np.mean(score_base_model_test_list),
            'runtime': np.mean(runtime_list),
            'parameter_setting': parameter_setting,
        }

    if config['computation']['verbose'] > 0:
        print('Number of Trials CART: ' + str(config['computation']['search_iterations']))

    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
        hpo_mode = "max" if metric in ['r2', 'accuracy', 'f1', 'roc_auc'] else "min"
        optuna_search = OptunaSearch(metric="cv_score", mode=hpo_mode, points_to_evaluate=[], seed=config['computation']['random_seed'])

        config_ray = {
                      'parameters': parameter_dict,
                      'dataset_dict_list': dataset_dict_list, 
                      'config': config,
                     }

        config['computation']['num_gpu'] = 0

        print_intermediate_tables = True if config['computation']['verbose'] > 2 else False 
        progress_reporter = JupyterNotebookReporter(overwrite=False,metric_columns=['cv_score','test_score','runtime'],max_progress_rows=config['computation']['search_iterations'],print_intermediate_tables=False,max_column_length=50) if config['computation']['verbose'] > 0 else JupyterNotebookReporter(overwrite=False, max_progress_rows=0, metric_columns=[], parameter_columns=[])

        analysis = tune.run(
                            evaluate_parameter_setting_CART,
                            config=config_ray,
                            search_alg=optuna_search,
                            resources_per_trial={"cpu": config['computation']['n_jobs'], 'gpu': config['computation']['num_gpu']},
                            num_samples=config['computation']['search_iterations'],
                            #log_to_file=False,
                            local_dir='./results_ray/' + timestr + '/',
                            #callbacks=[],
                            raise_on_failed_trial=False,        
                            verbose=config['computation']['verbose'],
                            progress_reporter=progress_reporter, #,parameter_columns=['cv_score','test_score','runtime']        
                            #progress_reporter=CustomProgressReporter(metric_columns=['cv_score','test_score','runtime']),
                            #keep_checkpoints_num=1,
                            #checkpoint_score_attr='cv_score'
                           )    

    grid_search_results = [tuple(trial.last_result.values()) for trial in analysis.trials]          
           
    replace_value = -np.inf if greater_better else np.inf

    replace_dict = {}
    for some_dict in [result[3] for result in grid_search_results]:
        if isinstance(some_dict, dict):
            for key in some_dict.keys():
                replace_dict[key] = None 
            break

    grid_search_results = [[np.nan_to_num(pd.to_numeric(result[0], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[1], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[2], errors='coerce'), nan=replace_value), 
                            result[3] if isinstance(result[3], dict) else replace_dict] for result in grid_search_results]
    
    cv_scores_base_model = np.nan_to_num([result[0] for result in grid_search_results], nan=replace_value)
    test_scores_base_model = np.nan_to_num([result[1] for result in grid_search_results], nan=replace_value)
    runtime_base_model = np.nan_to_num([result[2] for result in grid_search_results], nan=np.inf)
    parameter_settings = [result[3] for result in grid_search_results]


    grid_search_results_sorted = sorted(grid_search_results, key=lambda tup: tup[0], reverse=greater_better) 

    best_score_cv = grid_search_results_sorted[0][0]
    best_score_test = grid_search_results_sorted[0][1]
    best_model_runtime_cv = grid_search_results_sorted[0][2]
    best_params = grid_search_results_sorted[0][3]
   
    if config['GRANDE']['objective'] == 'classification':
        model_class = DecisionTreeClassifier
        model_identifier = 'CART_class'
    else:
        model_class = DecisionTreeRegressor      
        model_identifier = 'CART_reg'
            
    if config['computation']['report_test_performance']:
        with_valid = False
        encoded = True
                
        runtime_list = []
        base_model_list = []           
        dataset_dict_list = ray.get(dataset_dict_list)
        for i in range(config['computation']['cv_num_eval']):
            try:
                dataset_dict = dataset_dict_list[i][0]
            except:
                dataset_dict = dataset_dict_list[i]
                
            ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

            number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))
            
            base_model = model_class()
            base_model.set_params(**best_params)

            start = timeit.default_timer()
            if config['GRANDE']['class_weights']:
                base_model.fit(enforce_numpy(X_train), 
                               enforce_numpy(y_train),
                               sample_weight=calculate_sample_weights(enforce_numpy(y_train)))       
            else:
                base_model.fit(enforce_numpy(X_train), 
                               enforce_numpy(y_train))       
                
            end = timeit.default_timer()  
            runtime = end - start
            
            runtime_list.append(runtime)
            base_model_list.append(base_model)
        
    else:
        runtime_list = [best_model_runtime_cv for _ in range(config['computation']['cv_num_eval'])]
        base_model_list = [None for _ in range(config['computation']['cv_num_eval'])]
    
    if config['computation']['verbose'] > 0:
        print('Best Params CART')
        display(best_params)        
        print('Best CV-Score CART:', best_score_cv)
    
    hpo_path_by_dataset =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/' + identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/').mkdir(parents=True, exist_ok=True)    
        
    if not os.path.isfile(hpo_path_by_dataset):
        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()        
    
    hpo_path_by_dataset_with_timestamp =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/' + model_identifier + '___' +  identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/').mkdir(parents=True, exist_ok=True)    
    
    if not os.path.isfile(hpo_path_by_dataset_with_timestamp):
        file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()         
    
    return (best_score_cv, runtime_list, base_model_list), cv_scores_base_model, test_scores_base_model

def hpo_RandomForest(identifier, dataset_dict_list, parameter_dict, config, metric='f1', greater_better=True, timestr = 'placeholder'):
    
    def evaluate_parameter_setting_RandomForest(trial_config):   
        
        with_valid = False
        encoded = True     
        
        parameter_setting = trial_config['parameters']
        dataset_dict_list = ray.get(trial_config['dataset_dict_list'])
        config = trial_config['config']
        
        score_base_model_list = []
        runtime_list = []     
        
        if config['GRANDE']['objective'] == 'classification':
             model_class = RandomForestClassifier
        else:
             model_class = RandomForestRegressor  
                            
        for i in range(config['computation']['cv_num_eval']):   
            
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_cv_list = []
                runtime_list_cv = []                  
                for j in range(config['computation']['cv_num_hpo']):

                    dataset_dict = dataset_dict_list[i][j]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded, cv=True)

                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))

                    base_model = model_class()
                    base_model.set_params(**parameter_setting)

                    start = timeit.default_timer()
                    if config['GRANDE']['class_weights']:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train),
                                       sample_weight=calculate_sample_weights(enforce_numpy(y_train)))
                    else:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train))                        

                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)

                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)

                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_cv = 0.5
                    
                    else:
                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_cv_list.append(score_base_model_cv)
                    runtime_list_cv.append(runtime)

                score_base_model = np.mean(score_base_model_cv_list)

                score_base_model_list.append(score_base_model)
                runtime_list.append(np.mean(runtime_list_cv))
            else:

                dataset_dict = dataset_dict_list[i]
                
                ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))      

                base_model = model_class()
                base_model.set_params(**parameter_setting)

                start = timeit.default_timer()
                if config['GRANDE']['class_weights']:
                    base_model.fit(enforce_numpy(X_train), 
                                   enforce_numpy(y_train),
                                   sample_weight=calculate_sample_weights(enforce_numpy(y_train)))
                else:
                    base_model.fit(enforce_numpy(X_train), 
                                   enforce_numpy(y_train))
                    
                end = timeit.default_timer()  
                runtime = end - start                

                base_model_pred = base_model.predict(enforce_numpy(X_test))
                base_model_pred = np.nan_to_num(base_model_pred)

                if config['GRANDE']['objective'] == 'classification':
                    base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                    base_model_pred_proba = np.nan_to_num(base_model_pred_proba)

                    if metric not in ['f1', 'roc_auc', 'accuracy']:
                        score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))
                    else:
                        if metric == 'f1':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                        if metric == 'accuracy':
                            score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                                
                        elif metric == 'roc_auc':
                            try:
                                if number_of_classes > 2:                            
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                else:
                                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                            except ValueError:
                                score_base_model = 0.5
                else:
                    score_base_model = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                score_base_model_list.append(score_base_model)
                runtime_list.append(runtime)                
                
        
        if config['computation']['report_test_performance']:
            if config['computation']['cv_num_hpo'] >= 1:
                score_base_model_test_list = []
                runtime_test_list = []                       

                for i in range(config['computation']['cv_num_eval']):   

                    dataset_dict = dataset_dict_list[i][0]
                    
                    ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

                    number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))              

                    base_model = model_class()
                    base_model.set_params(**parameter_setting)

                    start = timeit.default_timer()
                    if config['GRANDE']['class_weights']:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train),
                                       sample_weight=calculate_sample_weights(enforce_numpy(y_train)))
                    else:
                        base_model.fit(enforce_numpy(X_train), 
                                       enforce_numpy(y_train))
                    end = timeit.default_timer()  
                    runtime = end - start                

                    base_model_pred = base_model.predict(enforce_numpy(X_test))
                    base_model_pred = np.nan_to_num(base_model_pred)

                    if config['GRANDE']['objective'] == 'classification':
                        base_model_pred_proba = base_model.predict_proba(enforce_numpy(X_test))
                        base_model_pred_proba = np.nan_to_num(base_model_pred_proba)

                        if metric not in ['f1', 'roc_auc', 'accuracy']:
                            score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), base_model_pred_proba)
                        else:
                            if metric == 'f1':
                                score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred), average='macro')
                            if metric == 'accuracy':
                                score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))                       
                            elif metric == 'roc_auc':
                                try:
                                    if number_of_classes > 2:                            
                                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(tf.keras.utils.to_categorical(base_model_pred_proba, num_classes=number_of_classes)), multi_class='ovo')
                                    else:
                                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred_proba))       

                                except ValueError:
                                    score_base_model_test = 0.5

                    else:
                        score_base_model_test = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_test), np.round(base_model_pred))

                    score_base_model_test_list.append(score_base_model_test)
                    runtime_test_list.append(runtime)
      
            else:
                score_base_model_test_list = score_base_model_list
        else:
            score_base_model_test_list = score_base_model_list                
        
        return {
            'cv_score': np.mean(score_base_model_list),
            'test_score': np.mean(score_base_model_test_list),
            'runtime': np.mean(runtime_list),
            'parameter_setting': parameter_setting,
        }

    if config['computation']['verbose'] > 0:
        print('Number of Trials RandomForest: ' + str(config['computation']['search_iterations']))
    
    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
        hpo_mode = "max" if metric in ['r2', 'accuracy', 'f1', 'roc_auc'] else "min"
        optuna_search = OptunaSearch(metric="cv_score", mode=hpo_mode, points_to_evaluate=[], seed=config['computation']['random_seed'])    

        config_ray = {
                      'parameters': parameter_dict,
                      'dataset_dict_list': dataset_dict_list, 
                      'config': config,
                     }

        config['computation']['num_gpu'] = 0

        print_intermediate_tables = True if config['computation']['verbose'] > 2 else False 
        progress_reporter = JupyterNotebookReporter(overwrite=False,metric_columns=['cv_score','test_score','runtime'],max_progress_rows=config['computation']['search_iterations'],print_intermediate_tables=False,max_column_length=50) if config['computation']['verbose'] > 0 else JupyterNotebookReporter(overwrite=False, max_progress_rows=0, metric_columns=[], parameter_columns=[])

        analysis = tune.run(
                            evaluate_parameter_setting_RandomForest,
                            config=config_ray,
                            search_alg=optuna_search,
                            resources_per_trial={"cpu": config['computation']['n_jobs'], 'gpu': config['computation']['num_gpu']},
                            num_samples=config['computation']['search_iterations'],
                            #log_to_file=False,
                            local_dir='./results_ray/' + timestr + '/',
                            #callbacks=[],
                            raise_on_failed_trial=False,        
                            verbose=config['computation']['verbose'],
                            progress_reporter=progress_reporter, #,parameter_columns=['cv_score','test_score','runtime']        
                            #progress_reporter=CustomProgressReporter(metric_columns=['cv_score','test_score','runtime']),
                            #keep_checkpoints_num=1,
                            #checkpoint_score_attr='cv_score'
                           )    

    grid_search_results = [tuple(trial.last_result.values()) for trial in analysis.trials]          
           
    replace_value = -np.inf if greater_better else np.inf

    replace_dict = {}
    for some_dict in [result[3] for result in grid_search_results]:
        if isinstance(some_dict, dict):
            for key in some_dict.keys():
                replace_dict[key] = None 
            break

    grid_search_results = [[np.nan_to_num(pd.to_numeric(result[0], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[1], errors='coerce'), nan=replace_value), 
                            np.nan_to_num(pd.to_numeric(result[2], errors='coerce'), nan=replace_value), 
                            result[3] if isinstance(result[3], dict) else replace_dict] for result in grid_search_results]
    
    cv_scores_base_model = np.nan_to_num([result[0] for result in grid_search_results], nan=replace_value)
    test_scores_base_model = np.nan_to_num([result[1] for result in grid_search_results], nan=replace_value)
    runtime_base_model = np.nan_to_num([result[2] for result in grid_search_results], nan=np.inf)
    parameter_settings = [result[3] for result in grid_search_results]


    grid_search_results_sorted = sorted(grid_search_results, key=lambda tup: tup[0], reverse=greater_better) 

    best_score_cv = grid_search_results_sorted[0][0]
    best_score_test = grid_search_results_sorted[0][1]
    best_model_runtime_cv = grid_search_results_sorted[0][2]
    best_params = grid_search_results_sorted[0][3]
   
    if config['GRANDE']['objective'] == 'classification':
        model_class = RandomForestClassifier
        model_identifier = 'RF_class'
    else:
        model_class = RandomForestRegressor      
        model_identifier = 'RF_reg'
            
    if config['computation']['report_test_performance']:
        with_valid = False
        encoded = True     
                
        runtime_list = []
        base_model_list = []           
        dataset_dict_list = ray.get(dataset_dict_list)
        for i in range(config['computation']['cv_num_eval']):
            try:
                dataset_dict = dataset_dict_list[i][0]
            except:
                dataset_dict = dataset_dict_list[i]
                
            ((X_train, y_train), (X_valid, y_valid),(X_test, y_test)) = get_datasets_from_dataset_dict(dataset_dict, with_valid=with_valid, encoded=encoded)

            number_of_classes=len(np.unique(pd.concat([y_train, y_valid, y_test])))
            
            base_model = model_class()
            base_model.set_params(**best_params)

            start = timeit.default_timer()
            if config['GRANDE']['class_weights']:
                base_model.fit(enforce_numpy(X_train), 
                               enforce_numpy(y_train),
                               sample_weight=calculate_sample_weights(enforce_numpy(y_train)))   
            else:
                base_model.fit(enforce_numpy(X_train), 
                               enforce_numpy(y_train))                   
            end = timeit.default_timer()  
            runtime = end - start
            
            runtime_list.append(runtime)
            base_model_list.append(base_model)
        
    else:
        runtime_list = [best_model_runtime_cv for _ in range(config['computation']['cv_num_eval'])]
        base_model_list = [None for _ in range(config['computation']['cv_num_eval'])]
    
    if config['computation']['verbose'] > 0:
        print('Best Params RandomForest')
        display(best_params)
        print('Best CV-Score RandomForest:', best_score_cv)

    hpo_path_by_dataset =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/' + identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/').mkdir(parents=True, exist_ok=True)    
        
    if not os.path.isfile(hpo_path_by_dataset):
        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()        
    
    hpo_path_by_dataset_with_timestamp =  './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/' + model_identifier + '___' +  identifier + '.csv'
    Path('./evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/').mkdir(parents=True, exist_ok=True)    
    
    if not os.path.isfile(hpo_path_by_dataset_with_timestamp):
        file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['cv_score', 'test_score', 'runtime', list(best_params.keys())]))

    file_by_dataset = open(hpo_path_by_dataset_with_timestamp, 'a+')
    writer = csv.writer(file_by_dataset)         
    for score_cv, score_test, runtime, params in zip(cv_scores_base_model, test_scores_base_model, runtime_base_model, parameter_settings):
        writer.writerow(flatten_list([score_cv, score_test, runtime, list(params.values())]))
    file_by_dataset.close()           
    
    return (best_score_cv, runtime_list, base_model_list), cv_scores_base_model, test_scores_base_model


def structure_evaluation_results(evaluation_results, 
                                 benchmark_dict,
                                 identifier_list,
                                 config,
                                 metrics = [],
                                 metric_identifer='test',
                                 smaller_better=False):
    
    smaller_better_names = ['neg_mean_absolute_percentage_error', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'Runtime']
    
    for i, result in enumerate(evaluation_results):
        if i == 0:
            model_dict = result[0]
            scores_dict = result[1]
            dataset_dict = result[2]
        else: 
            model_dict = mergeDict(model_dict, result[0])
            scores_dict = mergeDict(scores_dict, result[1])
            dataset_dict = mergeDict(dataset_dict, result[2])    

    model_identifier_list = flatten_list(['GRANDE', list(benchmark_dict.keys())])           
            
    metric_identifer = '_' + metric_identifer
    index = identifier_list
    columns = flatten_list([[[approach + ' ' + metric + '_mean', approach + ' ' + metric + '_max', approach + ' ' + metric + '_std'] for metric in metrics] for approach in model_identifier_list])

    
    results_list = []
    runtime_list = []
    for model_identifier in model_identifier_list:
        results = None
        
        for metric in metrics:
            scores = [scores_dict[identifier][model_identifier][metric + metric_identifer] for identifier in identifier_list]

            scores_mean = np.mean(scores, axis=1) if config['computation']['cv_num_eval'] > 1 else scores

            scores_max = np.max(scores, axis=1) if config['computation']['cv_num_eval'] > 1 else scores

            scores_std = np.std(scores, axis=1) if config['computation']['cv_num_eval'] > 1 else np.array([0.0] * len(scores))

            results_by_metric = np.vstack([scores_mean, scores_max, scores_std])

            if results is None:
                results = results_by_metric        
            else:
                results = np.vstack([results, results_by_metric])             
                
        results_list.append(results)
        
        runtimes = np.array([scores_dict[identifier][model_identifier]['runtime'] for identifier in identifier_list])
        runtime_list.append(runtimes)          
            
    scores_dataframe = pd.DataFrame(data=np.vstack(results_list).T, index = index, columns = columns)     
 
    columns = flatten_list([[model_identifier + ' ' + measure for model_identifier in model_identifier_list] for measure in ['Mean', 'Std', 'Max']])
    if config['computation']['cv_num_eval'] > 1:
        
        runtime_results = pd.DataFrame(data=np.vstack([[[np.round(np.mean(np.genfromtxt(runtimes_by_data.astype(str))), 3) if not np.isnan(np.genfromtxt(runtimes_by_data.astype(str))).any() else np.unique([entry for entry in runtimes_by_data if np.isnan(np.genfromtxt(np.array([entry]).astype(str)))])[0] for runtimes_by_data in runtimes] for runtimes in runtime_list], 
                                                                          [[np.round(np.std(np.genfromtxt(runtimes_by_data.astype(str)))) if not np.isnan(np.genfromtxt(runtimes_by_data.astype(str))).any() else np.unique([entry for entry in runtimes_by_data if np.isnan(np.genfromtxt(np.array([entry]).astype(str)))])[0] for runtimes_by_data in runtimes] for runtimes in runtime_list], 
                                                                          [[np.round(np.max(np.genfromtxt(runtimes_by_data.astype(str)))) if not np.isnan(np.genfromtxt(runtimes_by_data.astype(str))).any() else np.unique([entry for entry in runtimes_by_data if np.isnan(np.genfromtxt(np.array([entry]).astype(str)))])[0] for runtimes_by_data in runtimes] for runtimes in runtime_list]]).T, index=identifier_list, columns=columns)        
    
    else:
        runtime_results = pd.DataFrame(data=np.vstack([[runtimes for runtimes in runtime_list], 
                                                                          [np.array([0.0] * runtimes.shape[0]) for runtimes in runtime_list], 
                                                                          [runtimes for runtimes in runtime_list]]).T, index=identifier_list, columns=columns)
        
        

        
        
    index = [index_name.split(' ')[1] for index_name in scores_dataframe.mean()[:scores_dataframe.shape[1]//len(model_identifier_list)].index]
    mean_result_dataframe = np.round(pd.DataFrame(data=np.vstack(np.array_split(scores_dataframe.mean(), len(model_identifier_list))).T, index=index, columns=model_identifier_list), 3)
               
    mean_runtime_results = np.round(pd.DataFrame(data=np.vstack(np.array_split(runtime_results.apply(pd.to_numeric, errors='coerce').mean(), 3))[[0,2,1]], index=['runtime_mean', 'runtime_max', 'runtime_std'], columns=model_identifier_list), 3)
    
    mean_result_dataframe = pd.concat([mean_result_dataframe, mean_runtime_results])
    
    
    
    scores_dataframe_mean = scores_dataframe.mean()
    scores_dataframe_mean.name = 'MEAN'
    count_list = []
    for column in scores_dataframe.columns:
        column_metric_identifier = column.split(' ')[1]
        if smaller_better:
            count_series = scores_dataframe[column]<=get_columns_by_name(scores_dataframe, column_metric_identifier).min(axis=1)
        else:
            count_series = scores_dataframe[column]>=get_columns_by_name(scores_dataframe, column_metric_identifier).max(axis=1)
        #count_series.drop('MEAN', inplace=True)
        count = count_series.sum()
        count_list.append(count)
    scores_dataframe_count = pd.Series(count_list, index = scores_dataframe.columns)
    scores_dataframe_count.name = 'COUNT'    
    
    mean_reciprocal_rank_list_complete = []
    for i, row in scores_dataframe.iterrows():
        row_reciprocal_ranks = []
        
        num_row_types = len(['mean', 'max', 'std'])*len(metrics)
        for score_type_identifier in range(num_row_types):
            std_identifer = (score_type_identifier + 1) % len(['mean', 'max', 'std']) == 0
            
            index_change = -2 if std_identifer else 0 
            if smaller_better:
                sorted_row = np.sort(row[score_type_identifier+index_change::num_row_types].values)
            else:
                sorted_row = np.sort(row[score_type_identifier+index_change::num_row_types].values)[::-1]
            
            sorted_row =  sorted_row[~np.isnan(sorted_row)]
            row_reciprocal_ranks_identifier = []
            for score in row[score_type_identifier+index_change::num_row_types].values:
                rank = np.where(sorted_row == score)[0][0] + 1 if score in sorted_row else np.nan
                reciprocal_rank = 1/rank if not np.isnan(rank) else np.nan
                row_reciprocal_ranks_identifier.append(reciprocal_rank)
            row_reciprocal_ranks.append(row_reciprocal_ranks_identifier)
            
        row_reciprocal_ranks = np.hstack(np.hstack(np.dstack(row_reciprocal_ranks)))
        mean_reciprocal_rank_list_complete.append(row_reciprocal_ranks)
    
    
    mean_reciprocal_rank_list_complete = np.vstack(mean_reciprocal_rank_list_complete)
    mean_reciprocal_rank_list = np.nanmean(mean_reciprocal_rank_list_complete, axis=0)
    mean_reciprocal_rank_std_list = np.nanstd(mean_reciprocal_rank_list_complete, axis=0)
    
    mean_reciprocal_rank_list_combined = []
    for columnname, mean_reciprocal_rank, mean_reciprocal_rank_std in zip(scores_dataframe.columns, mean_reciprocal_rank_list, mean_reciprocal_rank_std_list):
        if 'std' in columnname:
            mean_reciprocal_rank_list_combined.append(mean_reciprocal_rank_std)
        else:
            mean_reciprocal_rank_list_combined.append(mean_reciprocal_rank)
    
    #print(mean_reciprocal_rank_list)
    scores_dataframe_reciprocal_rank = pd.Series(mean_reciprocal_rank_list_combined, index = scores_dataframe.columns)
    scores_dataframe_reciprocal_rank.name = 'MEAN RECIPROCAL RANK'
    
    
    scores_dataframe = scores_dataframe.append(scores_dataframe_mean)              
    scores_dataframe = scores_dataframe.append(scores_dataframe_count)         
    scores_dataframe = scores_dataframe.append(scores_dataframe_reciprocal_rank)         

    runtime_results_mean = runtime_results.mean()
    runtime_results_mean.name = 'MEAN'
    
    count_list = []
    for column in runtime_results.columns:
        column_metric_identifier = column.split(' ')[1]
        if smaller_better:
            count_series = np.genfromtxt(runtime_results[column].astype(str))<=np.genfromtxt(get_columns_by_name(runtime_results, column_metric_identifier).min(axis=1).astype(str)) if not np.isnan(np.genfromtxt(runtime_results[column].astype(str))).any() else np.genfromtxt(runtime_results[column].astype(str))

        else:
            count_series = np.genfromtxt(runtime_results[column].astype(str))>=np.genfromtxt(get_columns_by_name(runtime_results, column_metric_identifier).max(axis=1).astype(str)) if not np.isnan(np.genfromtxt(runtime_results[column].astype(str))).any() else np.genfromtxt(runtime_results[column].astype(str))

        
        count_series = np.genfromtxt(runtime_results[column].astype(str))>=np.genfromtxt(get_columns_by_name(runtime_results, column_metric_identifier).max(axis=1).astype(str)) if not np.isnan(np.genfromtxt(runtime_results[column].astype(str))).any() else np.genfromtxt(runtime_results[column].astype(str))
        count = count_series.sum()
        count_list.append(count)
    runtime_results_count = pd.Series(count_list, index = runtime_results.columns)
    runtime_results_count.name = 'COUNT'    
    
    mean_reciprocal_rank_list_complete = []
    sort_order = flatten_list([[i+j*runtime_results.shape[1]//len(['mean', 'max', 'std']) for j in [0,2,1]] for i in range(runtime_results.shape[1]//len(['mean', 'max', 'std']))])
    runtime_results = runtime_results.iloc[:,sort_order]
    runtime_results = runtime_results.apply(pd.to_numeric, errors='coerce')
    for i, row in runtime_results.iterrows():
        row_reciprocal_ranks = []
        
        num_row_types = len(['mean', 'max', 'std'])
        for score_type_identifier in range(num_row_types):
            std_identifer = (score_type_identifier + 1) % len(['mean', 'max', 'std']) == 0
            
            index_change = -2 if std_identifer else 0 
            sorted_row = np.sort(row[score_type_identifier+index_change::num_row_types].values)[::-1]
            sorted_row =  sorted_row[~np.isnan(sorted_row)]
            row_reciprocal_ranks_identifier = []
            for score in row[score_type_identifier+index_change::num_row_types].values:
                rank = np.where(sorted_row == score)[0][0] + 1 if score in sorted_row else np.nan
                reciprocal_rank = 1/rank if not np.isnan(rank) else np.nan
                row_reciprocal_ranks_identifier.append(reciprocal_rank)
            row_reciprocal_ranks.append(row_reciprocal_ranks_identifier)
            
        row_reciprocal_ranks = np.hstack(np.hstack(np.dstack(row_reciprocal_ranks)))
        mean_reciprocal_rank_list_complete.append(row_reciprocal_ranks)
    
    
    mean_reciprocal_rank_list_complete = np.vstack(mean_reciprocal_rank_list_complete)
    mean_reciprocal_rank_list = np.nanmean(mean_reciprocal_rank_list_complete, axis=0)
    mean_reciprocal_rank_std_list = np.nanstd(mean_reciprocal_rank_list_complete, axis=0)
    
    mean_reciprocal_rank_list_combined = []
    for columnname, mean_reciprocal_rank, mean_reciprocal_rank_std in zip(runtime_results.columns, mean_reciprocal_rank_list, mean_reciprocal_rank_std_list):
        if 'std' in columnname:
            mean_reciprocal_rank_list_combined.append(mean_reciprocal_rank_std)
        else:
            mean_reciprocal_rank_list_combined.append(mean_reciprocal_rank)
    
    #print(mean_reciprocal_rank_list)
    runtime_results_reciprocal_rank = pd.Series(mean_reciprocal_rank_list_combined, index = runtime_results.columns)
    runtime_results_reciprocal_rank.name = 'MEAN RECIPROCAL RANK'    
    
    
    runtime_results = runtime_results.append(runtime_results_mean)          
    runtime_results = runtime_results.append(runtime_results_count)  
    runtime_results = runtime_results.append(runtime_results_reciprocal_rank)         
    
    return scores_dataframe, runtime_results, mean_result_dataframe
        

    
    
def get_hpo_best_params_by_dataset(timestr, dataset_name):
    
    for depth in range(1, 20):
        try:
            #filepath = './evaluation_results' + config['computation']['hpo_path'] + '/depth' + str(config['GRANDE']['depth']) + '/' + timestr + '/'
            filepath = './evaluation_results' + config['computation']['hpo_path'] + '/depth' + str(depth) + '/' + timestr + '/'
            if 'BIN' in dataset_name:
                with open(filepath + 'hpo_best_params_classification_binary.pickle', 'rb') as file:
                    hpo_best_params = pickle.load(file, protocol=pickle.HIGHEST_PROTOCOL)
            elif 'MULT' in dataset_name:
                with open(filepath + 'hpo_best_params_classification_multi.pickle', 'rb') as file:
                    hpo_best_params = pickle.load(file, protocol=pickle.HIGHEST_PROTOCOL)    
            elif 'REG' in dataset_name:
                with open(filepath + 'hpo_best_params_classification_regression.pickle', 'rb') as file:
                    hpo_best_params = pickle.load(file, protocol=pickle.HIGHEST_PROTOCOL)       
        except:
            pass
        
    return hpo_best_params[dataset_name] 




def write_hpo_results_to_csv(hpo_results_real_world_cv, hpo_results_real_world_test, identifier_list, config, timestr):
        
    hpo_path = './evaluation_results' + config['computation']['hpo_path'] + '/hpo_best/GRANDE/'
    Path(hpo_path).mkdir(parents=True, exist_ok=True)    

    for identifier in identifier_list:
        hpo_path_by_dataset = hpo_path + identifier + '.csv'
        #file_by_dataset = open(hpo_path_by_dataset, 'r+')
        
        params_dicts_with_score = [(params_dict_cv['GRANDE mean (mean)'], params_dict_test['GRANDE mean (mean)'], params_dict_test['GRANDE runtime mean'], params_dict_cv) for params_dict_cv, params_dict_test in zip(hpo_results_real_world_cv[identifier], hpo_results_real_world_test[identifier])]
        params_dicts_with_score_sorted = sorted(params_dicts_with_score, key=lambda tup: tup[0], reverse=True)
        params_dicts_sorted = [dict_with_score[3] for dict_with_score in params_dicts_with_score_sorted]        
        
        best_params_dict = flatten_dict(params_dicts_sorted[0]['parameters_complete'])
        best_params_cv_score =  params_dicts_with_score_sorted[0][0]
        best_params_test_score =  params_dicts_with_score_sorted[0][1]
        best_params_runtime =  params_dicts_with_score_sorted[0][2]
        headers_by_dataset_dict = flatten_list_one_level(['cv_score', 'test_score', 'runtime', list(best_params_dict.keys())])#list(best_params_dict.keys())

        if os.path.isfile(hpo_path_by_dataset):
            file_by_dataset = open(hpo_path_by_dataset, 'r+')
            reader = csv.reader(file_by_dataset)
            headers_by_dataset_file = next(reader, None)

            counter = 1
            while not headers_by_dataset_dict == headers_by_dataset_file:
                hpo_path_by_dataset = hpo_path + identifier + str(counter) + '.csv'

                if not os.path.isfile(hpo_path_by_dataset):
                    file_by_dataset = open(hpo_path_by_dataset, 'a+')
                    writer = csv.writer(file_by_dataset)
                    writer.writerow(headers_by_dataset_dict)
                    break

                file_by_dataset = open(hpo_path_by_dataset, 'r+')

                reader = csv.reader(file_by_dataset)
                headers_by_dataset_file = next(reader, None)  
                counter += 1
        else:
            file_by_dataset = open(hpo_path_by_dataset, 'a+')
            writer = csv.writer(file_by_dataset)   
            writer.writerow(headers_by_dataset_dict)

        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)
        writer.writerow(flatten_list_one_level([best_params_cv_score, best_params_test_score, best_params_runtime, list(best_params_dict.values())]))
        file_by_dataset.close()    
    
    hpo_path = './evaluation_results' + config['computation']['hpo_path'] + '/hpo_complete/GRANDE/'
    Path(hpo_path).mkdir(parents=True, exist_ok=True)    

    for identifier in identifier_list:
        hpo_path_by_dataset = hpo_path + identifier + '.csv'
        #file_by_dataset = open(hpo_path_by_dataset, 'r+')

        params_dict_cv = hpo_results_real_world_cv[identifier]
        params_dict_test = hpo_results_real_world_test[identifier]
        for setting_number in range(len(params_dict_cv)):
        
            params_dict = flatten_dict(params_dict_cv[setting_number]['parameters_complete'])
            params_cv_score =  params_dict_cv[setting_number]['GRANDE mean (mean)']
            params_test_score =  params_dict_test[setting_number]['GRANDE mean (mean)']
            params_runtime =  params_dict_test[setting_number]['GRANDE runtime mean']
            headers_by_dataset_dict = flatten_list(['cv_score', 'test_score', 'runtime', list(params_dict.keys())])#list(best_params_dict.keys())

            if os.path.isfile(hpo_path_by_dataset):
                file_by_dataset = open(hpo_path_by_dataset, 'r+')
                reader = csv.reader(file_by_dataset)
                headers_by_dataset_file = next(reader, None)

                counter = 1
                while not headers_by_dataset_dict == headers_by_dataset_file:
                    hpo_path_by_dataset = hpo_path + identifier + str(counter) + '.csv'

                    if not os.path.isfile(hpo_path_by_dataset):
                        file_by_dataset = open(hpo_path_by_dataset, 'a+')
                        writer = csv.writer(file_by_dataset)
                        writer.writerow(headers_by_dataset_dict)
                        break

                    file_by_dataset = open(hpo_path_by_dataset, 'r+')

                    reader = csv.reader(file_by_dataset)
                    headers_by_dataset_file = next(reader, None)  
                    counter += 1
            else:
                file_by_dataset = open(hpo_path_by_dataset, 'a+')
                writer = csv.writer(file_by_dataset)   
                writer.writerow(headers_by_dataset_dict)

            file_by_dataset = open(hpo_path_by_dataset, 'a+')
            writer = csv.writer(file_by_dataset)
            writer.writerow(flatten_list_one_level([params_cv_score, params_test_score, params_runtime, list(params_dict.values())]))
            file_by_dataset.close()

    hpo_path = './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + timestr + '/GRANDE___'
    Path(hpo_path).mkdir(parents=True, exist_ok=True)    

    for identifier in identifier_list:
        hpo_path_by_dataset = hpo_path + identifier + '.csv'
        #file_by_dataset = open(hpo_path_by_dataset, 'r+')
        params_dict_cv = hpo_results_real_world_cv[identifier]
        params_dict_test = hpo_results_real_world_test[identifier]
        for setting_number in range(len(params_dict_cv)):
        
            params_dict = flatten_dict(params_dict_cv[setting_number]['parameters_complete'])
            params_cv_score =  params_dict_cv[setting_number]['GRANDE mean (mean)']
            params_test_score =  params_dict_test[setting_number]['GRANDE mean (mean)']
            params_runtime =  params_dict_test[setting_number]['GRANDE runtime mean']
            headers_by_dataset_dict = flatten_list(['cv_score', 'test_score', 'runtime', list(params_dict.keys())])#list(best_params_dict.keys())

            if os.path.isfile(hpo_path_by_dataset):
                file_by_dataset = open(hpo_path_by_dataset, 'r+')
                reader = csv.reader(file_by_dataset)
                headers_by_dataset_file = next(reader, None)

                counter = 1
                while not headers_by_dataset_dict == headers_by_dataset_file:
                    hpo_path_by_dataset = hpo_path + identifier + str(counter) + '.csv'

                    if not os.path.isfile(hpo_path_by_dataset):
                        file_by_dataset = open(hpo_path_by_dataset, 'a+')
                        writer = csv.writer(file_by_dataset)
                        writer.writerow(headers_by_dataset_dict)
                        break

                    file_by_dataset = open(hpo_path_by_dataset, 'r+')

                    reader = csv.reader(file_by_dataset)
                    headers_by_dataset_file = next(reader, None)  
                    counter += 1
            else:
                file_by_dataset = open(hpo_path_by_dataset, 'a+')
                writer = csv.writer(file_by_dataset)   
                writer.writerow(headers_by_dataset_dict)

            file_by_dataset = open(hpo_path_by_dataset, 'a+')
            writer = csv.writer(file_by_dataset)
            writer.writerow(flatten_list_one_level([params_cv_score, params_test_score, params_runtime, list(params_dict.values())]))
            file_by_dataset.close()            

def read_best_hpo_result_from_csv_benchmark(dataset_name,
                                            model_identifier, 
                                            config=None, 
                                            return_best_only=True, 
                                            ascending=False):
    hpo_path_by_dataset = './evaluation_results' + config['computation']['hpo_path'] + '/hpo/' + model_identifier + '/' + dataset_name  +  '.csv'    

    hpo_results = pd.read_csv(hpo_path_by_dataset) 
    #print(hpo_results.shape)
    #print(hpo_results.columns)
    #print(hpo_results.values[0])

    max_value = None
    if config['computation']['eval_metric_class'][0] in ['f1', 'accuracy', 'r2', 'balanced_accuracy']:
        max_value = 1
    
    if max_value is not None:
        hpo_results['cv_score'][hpo_results['cv_score'] > max_value] = -max_value


    hpo_results.sort_values(by=['cv_score'], ascending=ascending, inplace=True)    

    hpo_results_best_dict_with_index = hpo_results.iloc[:1].to_dict()

    hpo_results_best_dict = {'model': {}}
    for key, values in hpo_results_best_dict_with_index.items():
        if key == 'cv_score':
            hpo_results_best_dict[key] = list(values.values())[0]
        elif key == 'test_score':
            hpo_results_best_dict[key] = list(values.values())[0]
        elif key == 'runtime':
            hpo_results_best_dict[key] = list(values.values())[0]            
        else:
            try:
                hpo_results_best_dict['model'][key] = list(values.values())[0]
            except:
                hpo_results_best_dict['model'][key] = None

    if return_best_only:
        return hpo_results_best_dict

        
    
    return hpo_results_best_dict, hpo_results    
    

def write_latex_table_top(f):
    f.write('\\begin{table}[htb]' + '\n')
    f.write('\\centering' + '\n')
    f.write('\\resizebox{\columnwidth}{!}{' + '\n')
    f.write('%\\begin{threeparttable}' + '\n')

    f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' + '\n')
    
    
def write_latex_table_bottom(f, model_type):
    f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' + '\n')

    f.write('%\\begin{tablenotes}' + '\n')
    f.write('%\\item[a] \\footnotesize' + '\n')
    f.write('%\\item[b] \\footnotesize' + '\n')
    f.write('%\\end{tablenotes}' + '\n')
    f.write('%\\end{threeparttable}' + '\n')
    f.write('}' + '\n')
    f.write('\\caption{\\textbf{' + model_type +' Performance Comparison.} We report the train and test f1-score (mean $\pm$ stdev over 10 trials) and dataset specification. We also report the ranking of each approach for the corresponding dataset in brackets.}' + '\n')
    f.write('\\label{tab:eval-results_' + model_type.split(' ')[0] + '}' + '\n')
    f.write('\\end{table}' + '\n')
    
def add_hline(latex: str, index: int) -> str:
    """
    Adds a horizontal `index` lines before the last line of the table

    Args:
        latex: latex table
        index: index of horizontal line insertion (in lines)
    """
    lines = latex.splitlines()
    lines.insert(len(lines) - index - 2, r'\midrule')
    return '\n'.join(lines)#.replace('NaN', '')



def combine_mean_std_columns(dataframe, reverse=False, highlight=True):
    dataframe_combined_list = [] 
    column_name_list = []

    if reverse:
        mean_columns_comparator = np.argsort(np.argsort(dataframe.iloc[:,::2].values)) + 1
    else:
        mean_columns_comparator = np.argsort(np.argsort(-dataframe.iloc[:,::2].values)) + 1
    #dataframe.iloc[:,::2]

    index_list = []
    #display(dataframe)
    for index_name in dataframe.index[:-3]:
        index_list.append(index_name.split(':')[1])
    
    for column_index in range(len(dataframe.columns)//2):
        column_index_mean = 2*column_index
        column_index_std = 2*column_index+1

        column_mean = dataframe.iloc[:,column_index_mean]
        column_std = dataframe.iloc[:,column_index_std]

        column_name = column_mean.name.split(' ')[0]
        column_name_list.append(column_name)

        #combined_column = '$' + np.round(column_mean, 3).astype(str) + ' \pm ' +  np.round(column_std, 3).astype(str) + ' (' +  mean_columns_comparator[:,column_index].astype(str) + ')$'

        combined_column_list = []
        if mean_columns_comparator[:,0].shape[0] // 3 == 1:
            rank_list = mean_columns_comparator[:,column_index]
        else:
            rank_list_raw = mean_columns_comparator[:,column_index]
            rank_list = []
            for i in range(mean_columns_comparator[:,0].shape[0] // 3):
                if i == mean_columns_comparator[:,0].shape[0] // 3 - 1:
                    rank_list.append(np.argsort(np.argsort(-1 * rank_list_raw[i*3:(i+1)*3]))+1)     
                else:
                    rank_list.append(np.argsort(np.argsort(rank_list_raw[i*3:(i+1)*3]))+1)     

            rank_list = flatten_list(rank_list)        
        
        #display(column_std)
        #display(column_mean)
        for value_mean, value_std, rank in zip(column_mean, column_std, mean_columns_comparator[:,column_index]):
            if highlight:
                if rank == 1:
                    value = '\\bftab ' + '{:.3f}'.format(round(value_mean, 3))+ ' $\pm$ ' + '{:.3f}'.format(round(value_std, 3)) + ' (' + rank.astype(str) + ')'
                    #value = '$\mathbf{' + '{:.3f}'.format(round(value_mean, 3))+ ' \pm ' + '{:.3f}'.format(round(value_std, 3)) + ' (' + rank.astype(str) + ')}$'
                else:
                    value = '{:.3f}'.format(round(value_mean, 3)) + ' $\pm$ ' + '{:.3f}'.format(round(value_std, 3)) + ' (' + rank.astype(str) + ')'
                    #value = '$' + '{:.3f}'.format(round(value_mean, 3)) + ' \pm ' + '{:.3f}'.format(round(value_std, 3)) + ' (' + rank.astype(str) + ')$'
            else:
                #value = '{:.3f}'.format(round(value_mean, 3)) + ' ± ' + '{:.3f}'.format(round(value_std, 3)) + ' (' + rank.astype(str) + ')'
                value = '{:.3f}'.format(round(value_mean, 3)) + '  $\pm$ ' + '{:.3f}'.format(round(value_std, 3)) + ' (' + rank.astype(str) + ')'
            combined_column_list.append(value)
        dataframe_combined_list.append(combined_column_list)

    dataframe_combined = pd.DataFrame(data=np.array(dataframe_combined_list).T[:-3,:], columns=column_name_list, index=index_list)

    result_column_list = []

    if dataframe_combined.shape[1] == 1:
        rank_list = np.array([1])
    elif dataframe_combined.shape[1] // 3 == 1:
        rank_list = (np.argsort(np.argsort(-dataframe.loc['MEAN RECIPROCAL RANK'][::2].values)) + 1)
    else:
        rank_list_raw = (np.argsort(np.argsort(-dataframe.loc['MEAN RECIPROCAL RANK'][::2].values)) + 1)
        rank_list = []
        for i in range(dataframe_combined.shape[1] // 3):
            if i == dataframe_combined.shape[1] // 3 - 1:
                rank_list.append(np.argsort(np.argsort(-1 * rank_list_raw[i*3:(i+1)*3]))+1)     
            else:
                rank_list.append(np.argsort(np.argsort(rank_list_raw[i*3:(i+1)*3]))+1)     
                
        rank_list = flatten_list(rank_list)
    for result_value, result_std, result_rank in zip(dataframe.loc['MEAN RECIPROCAL RANK'][::2].values, dataframe.loc['MEAN RECIPROCAL RANK'][1::2].values, rank_list):
        if highlight:
            if result_rank == 1:
                result_column = '\\bftab ' + '{:0.3f}'.format(round(result_value, 3)) +  ' $\pm$ ' + '{:.3f}'.format(round(result_std, 3)) + ' (' + result_rank.astype(str) + ')'
                #result_column = '$\mathbf{' + '{:0.3f}'.format(round(result_value, 3)) + ' (' + result_rank.astype(str) + ')}$'
            else:
                result_column = '{:0.3f}'.format(round(result_value, 3)) +  ' $\pm$ ' + '{:.3f}'.format(round(result_std, 3)) + ' (' + result_rank.astype(str) + ')'   
                #result_column = '${:0.3f}'.format(round(result_value, 3)) + ' (' + result_rank.astype(str) + ')$'   
        else:
            result_column = '{:0.3f}'.format(round(result_value, 3)) +  ' $\pm$ ' + '{:.3f}'.format(round(result_std, 3)) + ' (' + result_rank.astype(str) + ')'   
            #result_column = '{:0.3f}'.format(round(result_value, 3)) + ' (' + result_rank.astype(str) + ')'   
        result_column_list.append(result_column)

    #result_column = '$' + np.array(['{:0.3f}'.format(round(x, 3)) for x in dataframe.loc['MEAN'][::2].values]).astype(object) + ' (' + (np.argsort(np.argsort(-dataframe.loc['MEAN'][::2].values)) + 1).astype(str) + ')$'
    result_column_pandas = pd.DataFrame(data=np.array([result_column_list]), columns=column_name_list, index=['Mean Reciprocal Rank'])
                
    dataframe_combined = dataframe_combined.append(result_column_pandas)
    
    return dataframe_combined    
    
def plot_table_save_results(benchmark_dict,
                            evaluation_results_real_world,
                            identifier_list,
                            #scores_dataframe_real_world,
                            #runtime_results,
                            #mean_result_dataframe_real_world,
                            #metrics,
                            identifier_string,     
                            filepath,               
                            config,
                            plot_runtime=False,
                            terminal_output=False):
    
    return_dataframe_dict = {}
    
    if (identifier_string.split('_')[0] == 'regression' and isinstance(config['computation']['eval_metric_reg'], list)) or (not identifier_string.split('_')[0] == 'regression' and isinstance(config['computation']['eval_metric_class'], list)):
        
        if identifier_string.split('_')[0] == 'regression':
            metrics = config['computation']['metrics_reg']
            select_metric_name_list = config['computation']['eval_metric_reg']
        else:
            metrics = config['computation']['metrics_class']
            select_metric_name_list = config['computation']['eval_metric_class']

        smaller_better_names = ['neg_mean_absolute_percentage_error', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'Runtime']
        
        
        for select_metric_name in select_metric_name_list:  

            smaller_better = True if select_metric_name in smaller_better_names else False

            (scores_dataframe_real_world, 
             runtime_results, 
             mean_result_dataframe_real_world) = structure_evaluation_results(evaluation_results = evaluation_results_real_world,
                                                                                     benchmark_dict = benchmark_dict,
                                                                                     identifier_list = identifier_list,
                                                                                     config = config,
                                                                                     metrics = metrics,
                                                                                     metric_identifer=identifier_string.split('_')[1],
                                                                                     smaller_better=smaller_better)    

            model_identifier_list = flatten_list(['GRANDE', list(benchmark_dict.keys())])           
            runtime_results.columns = [column + str(' Runtime') for column in runtime_results.columns]    

            scores_dataframe_real_world.to_csv(filepath + 'scores_dataframe_' + select_metric_name + '_' + identifier_string)
            runtime_results.to_csv(filepath + 'runtime_results_' + select_metric_name + '_' + identifier_string)
            mean_result_dataframe_real_world.to_csv(filepath + 'mean_result_dataframe_' + select_metric_name + '_' + identifier_string)       

            reorder = flatten_list([[(i*len(metrics))+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
            scores_dataframe_real_world_combined = pd.concat([scores_dataframe_real_world[scores_dataframe_real_world.columns[0::3]].iloc[:,reorder], runtime_results.iloc[:,:len(model_identifier_list)]], axis=1)
            #smaller_better_names = ['neg_mean_absolute_percentage_error_mean', 'neg_mean_absolute_percentage_error_mean', 'neg_mean_absolute_error_mean', 'neg_mean_squared_error_mean', 'Runtime']
            #smaller_better_name_by_index = [any(smaller_better_name in column for smaller_better_name in smaller_better_names) for column in scores_dataframe_real_world_combined.columns]
            highlight_index = list(scores_dataframe_real_world_combined.index)
            highlight_index.remove('MEAN')
            highlight_index.remove('COUNT')



            columnnames_select_metric = []
            for column in scores_dataframe_real_world.columns:
                if select_metric_name in column and 'max' not in column:
                    columnnames_select_metric.append(column)
    
            if not terminal_output: 
                if not any([metric in '\t'.join(columnnames_select_metric) for metric in smaller_better_names]):
                    display(scores_dataframe_real_world[columnnames_select_metric].style.apply(lambda x: ['background-color: #779ECB' if (v == np.mean(x[::2]))
                                                                                                   else 'background-color: #FF6961' if (v == min(x[::2]))
                                                                                                   else 'background-color: #9CC95C' if (v == max(x[::2]))
                                                                                                   else '' for i, v in enumerate(x)], axis = 1, subset=pd.IndexSlice[highlight_index, :]))
                else:
                    display(scores_dataframe_real_world[columnnames_select_metric].style.apply(lambda x: ['background-color: #779ECB' if (v == np.mean(x[::2]))
                                                                                                   else 'background-color: #FF6961' if (v == max(x[::2]))
                                                                                                   else 'background-color: #9CC95C' if (v == min(x[::2]))
                                                                                                   else '' for i, v in enumerate(x)], axis = 1, subset=pd.IndexSlice[highlight_index, :]))                

            return_dataframe_dict[select_metric_name] = scores_dataframe_real_world[columnnames_select_metric]
                
            timestr = filepath.split('/')[-2]

            with open("./evaluation_results" + config['computation']['hpo_path'] + "/latex_tables/" + timestr + "/latex_table_" + select_metric_name + '_' + identifier_string + ".tex", "a+") as f:
                write_latex_table_top(f)
                f.write(add_hline(combine_mean_std_columns(scores_dataframe_real_world[columnnames_select_metric]).to_latex(index=True, bold_rows=False, escape=False), 1))
                write_latex_table_bottom(f, identifier_string)
                f.write('\n\n')        

            
    
    else:
    
        if identifier_string.split('_')[0] == 'regression':
            metrics = config['computation']['metrics_reg']
            select_metric_name = config['computation']['eval_metric_reg']
        else:
            metrics = config['computation']['metrics_class']
            select_metric_name = config['computation']['eval_metric_class']

        smaller_better = True if select_metric_name in smaller_better_names else False

        (scores_dataframe_real_world, 
         runtime_results, 
         mean_result_dataframe_real_world) = structure_evaluation_results(evaluation_results = evaluation_results_real_world,
                                                                                 benchmark_dict = benchmark_dict,
                                                                                 identifier_list = identifier_list,
                                                                                 config = config,
                                                                                 metrics = metrics,
                                                                                 metric_identifer=identifier_string.split('_')[1],
                                                                                 smaller_better=smaller_better)    

        model_identifier_list = flatten_list(['GRANDE', list(benchmark_dict.keys())])           
        runtime_results.columns = [column + str(' Runtime') for column in runtime_results.columns]    

        scores_dataframe_real_world.to_csv(filepath + 'scores_dataframe_' + identifier_string)
        runtime_results.to_csv(filepath + 'runtime_results_' + identifier_string)
        mean_result_dataframe_real_world.to_csv(filepath + 'mean_result_dataframe_' + identifier_string)       

        reorder = flatten_list([[(i*len(metrics))+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        scores_dataframe_real_world_combined = pd.concat([scores_dataframe_real_world[scores_dataframe_real_world.columns[0::3]].iloc[:,reorder], runtime_results.iloc[:,:len(model_identifier_list)]], axis=1)
        smaller_better_names = ['neg_mean_absolute_percentage_error', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'Runtime']
        #smaller_better_names = ['neg_mean_absolute_percentage_error_mean', 'neg_mean_absolute_percentage_error_mean', 'neg_mean_absolute_error_mean', 'neg_mean_squared_error_mean', 'Runtime']
        #smaller_better_name_by_index = [any(smaller_better_name in column for smaller_better_name in smaller_better_names) for column in scores_dataframe_real_world_combined.columns]
        highlight_index = list(scores_dataframe_real_world_combined.index)
        highlight_index.remove('MEAN')
        highlight_index.remove('COUNT')



        columnnames_select_metric = []
        for column in scores_dataframe_real_world.columns:
            if select_metric_name in column and 'max' not in column:
                columnnames_select_metric.append(column)

        if not terminal_output:
            if not any([metric in '\t'.join(columnnames_select_metric) for metric in smaller_better_names]):
                display(scores_dataframe_real_world[columnnames_select_metric].style.apply(lambda x: ['background-color: #779ECB' if (v == np.mean(x[::2]))
                                                                                               else 'background-color: #FF6961' if (v == min(x[::2]))
                                                                                               else 'background-color: #9CC95C' if (v == max(x[::2]))
                                                                                               else '' for i, v in enumerate(x)], axis = 1, subset=pd.IndexSlice[highlight_index, :]))
            else:
                display(scores_dataframe_real_world[columnnames_select_metric].style.apply(lambda x: ['background-color: #779ECB' if (v == np.mean(x[::2]))
                                                                                               else 'background-color: #FF6961' if (v == max(x[::2]))
                                                                                               else 'background-color: #9CC95C' if (v == min(x[::2]))
                                                                                               else '' for i, v in enumerate(x)], axis = 1, subset=pd.IndexSlice[highlight_index, :]))

        return_dataframe_dict[select_metric_name] = scores_dataframe_real_world[columnnames_select_metric]
                
        timestr = filepath.split('/')[-2]

        with open("./evaluation_results" + config['computation']['hpo_path'] + "/latex_tables/" + timestr + "/latex_table_" + identifier_string + ".tex", "a+") as f:
            write_latex_table_top(f)
            f.write(add_hline(combine_mean_std_columns(scores_dataframe_real_world[columnnames_select_metric]).to_latex(index=True, bold_rows=False, escape=False), 1))
            write_latex_table_bottom(f, identifier_string)
            f.write('\n\n')        
 

    if plot_runtime:
        #reorder = flatten_list([[(i*3)+j for i in range(2)] for j in range(3)])
        #runtime_results_latex = runtime_results.iloc[:,:-(len(benchmark_dict)+1)].iloc[:,reorder]
        runtime_results_latex = runtime_results.iloc[:,[i for i in range(runtime_results.shape[1]) if not (i+1)%3 == 2]]
        if not terminal_output:
            display(runtime_results_latex.style.apply(lambda x: ['background-color: #779ECB' if (v == np.mean(x[::2]))
                                                                   else 'background-color: #FF6961' if (v == max(x[::2]))
                                                                   else 'background-color: #9CC95C' if (v == min(x[::2]))
                                                                   else '' for i, v in enumerate(x)], axis = 1, subset=pd.IndexSlice[highlight_index, :]))
        else:
            terminal_combined_df = combine_mean_std_columns(pd.concat([scores_dataframe_real_world[columnnames_select_metric], runtime_results_latex], axis=1, join="inner"), highlight=False)

            columnnames = terminal_combined_df.columns

            columnnames_new = []
            for i, columnname in enumerate(columnnames):
                if i >= len(columnnames)//2:
                    columnnames_new.append(columnname + ' Runtime')
                else:
                    columnnames_new.append(columnname)

            print(tabulate(terminal_combined_df, headers='keys', tablefmt='psql'))        


        with open("./evaluation_results" + config['computation']['hpo_path'] + "/latex_tables/" + timestr + "/latex_table_runtime_" + identifier_string + ".tex", "w+") as f:
            write_latex_table_top(f)
            f.write(add_hline(combine_mean_std_columns(runtime_results_latex, reverse=True).to_latex(index=True, bold_rows=False, escape=False), 1))
            write_latex_table_bottom(f, 'RUNTIME')
            f.write('\n\n')          
    
    path = filepath + 'scores_dataframe_complete' + '_' + identifier_string + '.pickle'
    with open(path, 'wb') as file:
        pickle.dump(return_dataframe_dict, file)        

    return return_dataframe_dict 


def plot_methods_performance_with_std(df):
    methods = []
    means = []
    stds = []

    # Collect all method names, means, and standard deviations
    for i in range(0, df.shape[1], 2):
        methods.append(df.columns[i].replace('_mean', ''))
        means.append(df.iloc[:, i])
        stds.append(df.iloc[:, i+1])

    # Create plot with larger dimensions
    plt.figure(figsize=(25, 10))

    # Define some line and marker styles
    linestyles = ['--', '-.', ':']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    
    # Create an iterator that produces different combinations of linestyles and markers
    styles = itertools.product(markers, linestyles)

    # X values are the dataset names (indices of the DataFrame)
    x = np.arange(df.shape[0])

    # For each method, plot the means and standard deviations with a different line/marker style
    for i, (marker, line) in zip(range(len(methods)), styles):
        y = means[i]
        e = stds[i]

        # Mean performance line plot with different line/marker styles
        plt.plot(x, y, linestyle=line, marker=marker, label=methods[i])

        # Standard deviation "tunnel" with increased transparency
        plt.fill_between(x, y-e, y+e, alpha=0.1)

    # Modify X tick labels: wrap the text after 25 characters
    labels = [textwrap.fill(label, 25) for label in df.index]
    
    # Set X ticks and labels with larger rotation
    plt.xticks(x, labels, rotation=90)

    # Add title and labels
    plt.title('Mean Performance and Standard Deviation of Methods by Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Mean Performance')

    # Add legend outside the plot
    plt.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot with a bit of extra space on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    plt.show()
    
    
def plot_methods_performance_diff(df):
    methods = []
    means = []

    # Collect all method names and means
    for i in range(0, df.shape[1], 2):
        methods.append(df.columns[i].replace('_mean', ''))
        means.append(df.iloc[:, i])

    # Calculate minimum performance for each dataset
    min_performance = np.min([means[i] for i in range(len(methods))], axis=0)

    # Calculate difference to the minimum performance
    differences = [means[i] - min_performance for i in range(len(methods))]

    # Create plot with larger dimensions
    plt.figure(figsize=(25, 10))

    # Define some line and marker styles
    linestyles = ['--', '-.', ':']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

    # Create an iterator that produces different combinations of linestyles and markers
    styles = itertools.product(markers, linestyles)

    # X values are the dataset names (indices of the DataFrame)
    x = np.arange(df.shape[0])

    # For each method, plot the performance differences with a different line/marker style
    for i, (marker, line) in zip(range(len(methods)), styles):
        y = differences[i]

        # Performance difference line plot with different line/marker styles
        plt.plot(x, y, linestyle=line, marker=marker, label=methods[i])

    # Modify X tick labels: wrap the text after 25 characters
    labels = [textwrap.fill(label, 25) for label in df.index]
    
    # Set X ticks and labels with larger rotation
    plt.xticks(x, labels, rotation=90)

    # Add title and labels
    plt.title('Performance Difference of Methods by Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Performance Difference to Best Model')

    
    
    # Add legend outside the plot
    plt.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot with a bit of extra space on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    plt.show()

    
def plot_dt_comparison(evaluation_results_real_world,
                      identifier_list,
                      identifier_string,
                      timestr,
                      config):
    
    plot_identifier = identifier_list[0]     
    
    model_dict = [evaluation_result_real_world[0] for evaluation_result_real_world in evaluation_results_real_world]
    scores_dict = [evaluation_result_real_world[1] for evaluation_result_real_world in evaluation_results_real_world]        
    dataset_dict = [evaluation_result_real_world[2] for evaluation_result_real_world in evaluation_results_real_world]         
    
    if True:
        best_identifier = None
        best_plot_index = None
        difference_best = -np.inf

        for identifier in identifier_list:
            for plot_index in range(config['computation']['trials']):
                difference_current = scores_dict[plot_index][identifier]['GRANDE']['f1_test'] - scores_dict[plot_index][identifier]['CART']['f1_test']
                if difference_current > difference_best:
                    difference_best = difference_current
                    best_identifier = identifier
                    best_plot_index = plot_index  

        print(best_identifier)
        print(best_plot_index)
        plot_index = best_plot_index#0
        identifier = best_identifier#identifier_list[0]
    elif True:
        best_index = 0
        
        differences_dict = {}
        for identifier in identifier_list:
            differences_list = []
            
            for plot_index in range(config['computation']['trials']):
                
                difference_current = scores_dict[plot_index][identifier]['GRANDE']['f1_test'] - scores_dict[plot_index][identifier]['CART']['f1_test']
                differences_list.append(-difference_current)

            differences_dict[identifier] = np.argsort(differences_list)

        identifier = plot_identifier
        plot_index = differences_dict[identifier][best_index]
        
        print(identifier)
        print(plot_index)            
        
    else:
        plot_index = 0
        identifier = plot_identifier
    print('F1 Score (GRANDE):\t\t', scores_dict[plot_index][identifier]['GRANDE']['f1_test'])
    print('Accuracy Score (GRANDE):\t', scores_dict[plot_index][identifier]['GRANDE']['accuracy_test'])
    print('ROC AUC Score (GRANDE):\t', scores_dict[plot_index][identifier]['GRANDE']['roc_auc_test'])
    plt.figure(figsize=(15,8))
    image = model_dict[plot_index][identifier]['GRANDE'].plot(normalizer_list=dataset_dict[plot_index][identifier]['normalizer_list'])
    display(image)
    
    print('F1 Score (CART):\t', scores_dict[plot_index][identifier]['CART']['f1_test'])
    print('Accuracy Score (CART):\t', scores_dict[plot_index][identifier]['CART']['accuracy_test'])
    print('ROC AUC Score (CART):\t', scores_dict[plot_index][identifier]['CART']['roc_auc_test'])
    plt.figure(figsize=(15,8))
    plot_tree(model_dict[plot_index][identifier]['CART'], fontsize=10) 
    plt.show()
        
    os.makedirs(os.path.dirname("./evaluation_results" + config['computation']['hpo_path'] + "/plots/" + timestr +"/"), exist_ok=True)
        
    filehandler = open('./evaluation_results' + config['computation']['hpo_path'] + '/plots/' + timestr + '/GRANDE_' + identifier + '_' + identifier_string + '.pickle', "wb")
    dill.dump(model_dict[plot_index][identifier]['GRANDE'], filehandler)
    filehandler.close()

    file = open('./evaluation_results' + config['computation']['hpo_path'] + '/plots/' + timestr + '/GRANDE_' + identifier + '_' + identifier_string  + '.pickle', 'rb')
    GRANDE_loaded = dill.load(file)
    file.close()

    filehandler = open('./evaluation_results' + config['computation']['hpo_path'] + '/plots/' + timestr + '/CART_' + identifier + '_' + identifier_string  + '.pickle', "wb")
    dill.dump(model_dict[plot_index][identifier]['CART'], filehandler)
    filehandler.close()

    file = open('./evaluation_results' + config['computation']['hpo_path'] + '/plots/' + timestr + '/CART_' + identifier + '_' + identifier_string  + '.pickle', 'rb')
    CART_loaded = dill.load(file)
    file.close()    
    
    filehandler = open('./evaluation_results' + config['computation']['hpo_path'] + '/plots/' + timestr + '/normalizer_list_' + identifier + '_' + identifier_string  + '.pickle', "wb")
    dill.dump(dataset_dict[plot_index][identifier]['normalizer_list'], filehandler)
    filehandler.close()

    file = open('./evaluation_results' + config['computation']['hpo_path'] + '/plots/' + timestr + '/normalizer_list_' + identifier + '_' + identifier_string  + '.pickle', 'rb')
    normalizer_list_loaded = dill.load(file)
    file.close()    
    
    filehandler = open('./evaluation_results' + config['computation']['hpo_path'] + '/plots/' + timestr + '/data_dict_' + identifier + '_' + identifier_string  + '.pickle', "wb")
    dill.dump(dataset_dict[plot_index][identifier], filehandler)
    filehandler.close()

    file = open('./evaluation_results' + config['computation']['hpo_path'] + '/plots/' + timestr + '/data_dict_' + identifier + '_' + identifier_string  + '.pickle', 'rb')
    data_dict_list_loaded = dill.load(file)
    file.close()    
     
    
def plot_methods_performance_norm(df):
    methods = []
    means = []

    # Collect all method names and means
    for i in range(0, df.shape[1], 2):
        methods.append(df.columns[i].replace('_mean', ''))
        means.append(df.iloc[:, i])

    # Calculate minimum and maximum performance for each dataset
    min_performance = np.min([means[i] for i in range(len(methods))], axis=0)
    max_performance = np.max([means[i] for i in range(len(methods))], axis=0)

    # Calculate normalized performance for each method
    norm_performance = [(means[i] - min_performance) / (max_performance - min_performance) for i in range(len(methods))]

    # Create plot with larger dimensions
    plt.figure(figsize=(25, 10))

    # Define some line and marker styles
    linestyles = ['--', '-.', ':']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

    # Create an iterator that produces different combinations of linestyles and markers
    styles = itertools.product(markers, linestyles)

    # X values are the dataset names (indices of the DataFrame)
    x = np.arange(df.shape[0])

    # For each method, plot the performance differences with a different line/marker style
    for i, (marker, line) in zip(range(len(methods)), styles):
        y = norm_performance[i]

        # Normalized performance line plot with different line/marker styles
        plt.plot(x, y, linestyle=line, marker=marker, label=methods[i])

    # Modify X tick labels: wrap the text after 25 characters
    labels = [textwrap.fill(label, 25) for label in df.index]
    
    # Set X ticks and labels with larger rotation
    plt.xticks(x, labels, rotation=90)

    # Add title and labels
    plt.title('Normalized Performance of Methods by Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Normalized Performance')

    # Add legend outside the plot
    plt.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot with a bit of extra space on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    plt.show()
    
def plot_methods_performance_norm_affine(df, quantile=0.1, clip_negative=True):

    methods = []
    means = []

    # Collect all method names and means
    for i in range(0, df.shape[1], 2):
        methods.append(df.columns[i].replace('_mean', ''))
        means.append(df.iloc[:, i])

    if clip_negative:
        means = [np.maximum(0, means[i]) for i in range(len(methods))]     
        
    # Calculate minimum and maximum performance for each dataset
    max_performance = np.max([means[i] for i in range(len(methods))], axis=0)
    # Calculate quantile performance for each dataset
    quantile_performance = np.quantile([means[i] for i in range(len(methods))], quantile, axis=0)
    # Optionally, clip negative scores to zero before normalization


    # Calculate normalized performance for each method
    norm_performance = [(means[i].values - quantile_performance) / (max_performance - quantile_performance) for i in range(len(methods))]
    # Create plot with larger dimensions
    plt.figure(figsize=(25, 10))

    # Define some line and marker styles
    linestyles = ['--', '-.', ':']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

    # Create an iterator that produces different combinations of linestyles and markers
    styles = itertools.product(markers, linestyles)

    # X values are the dataset names (indices of the DataFrame)
    x = np.arange(df.shape[0])

    # For each method, plot the performance differences with a different line/marker style
    for i, (marker, line) in zip(range(len(methods)), styles):
        y = norm_performance[i]

        # Normalized performance line plot with different line/marker styles
        plt.plot(x, y, linestyle=line, marker=marker, label=methods[i])

    # Modify X tick labels: wrap the text after 25 characters
    labels = [textwrap.fill(label, 25) for label in df.index]
    
    # Set X ticks and labels with larger rotation
    plt.xticks(x, labels, rotation=90)

    # Add title and labels
    plt.title('Normalized Performance of Methods by Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Normalized Performance')

    # Add legend outside the plot
    plt.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot with a bit of extra space on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    plt.show()

    
def get_benchmark_dict(config, eval_identifier):
    benchmark_dict = {}

    for key, value in config['benchmarks'].items():
        if value == True:
            if (key == 'GeneticTree' or key == 'DNDT') and eval_identifier == 'regression':
                pass
            else:
                benchmark_dict[key] = None         
    
    return benchmark_dict

    
def prepare_score_dict(config):
    scores_dict = {'GRANDE': {}}

    for key, value in config['benchmarks'].items():
        if value == True:
            if config['GRANDE']['objective'] == 'regression' and (key == 'GeneticTree' or key == 'DNDT'):
                pass
            else:
                scores_dict[key] = {}    
                
    return scores_dict
    
    
def binarize_data(X):  
    return_df = True
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        return_df = False
    
    columnnames_one_hot = []
    columnnames_discretize = []
    columnnames_binary = []
    for columnname in X:
        unique_values = len(np.unique(X[columnname]))
        
        if unique_values <= 2:
            columnnames_binary.append(columnname)
        elif unique_values < 25:
            columnnames_one_hot.append(columnname)
        else:
            columnnames_discretize.append(columnname)
            
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), columnnames_one_hot), ('discretize', KBinsDiscretizer(), columnnames_discretize)], remainder='passthrough', sparse_threshold=0)
    transformer.fit(X)        

    X_values = transformer.transform(X)
    X = pd.DataFrame(X_values, columns=transformer.get_feature_names())
    
    
    if return_df:
        return X, transformer
    else:
        return X.values, transformer
    
def get_binarize_transformer(X_train, X_test):
    return_df = True
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
        #X_valid = pd.DataFrame(X_valid)
        X_test = pd.DataFrame(X_test)
        return_df = False  
        
    #X = pd.concat([X_train, X_valid, X_test])
    X = pd.concat([X_train, X_test])
        
    columnnames_one_hot = []
    columnnames_discretize = []
    columnnames_binary = []
    for columnname in X:
        unique_values = len(np.unique(X[columnname]))
        if unique_values <= 2:
            columnnames_binary.append(columnname)
        elif unique_values < 10:
            columnnames_one_hot.append(columnname)
        else:
            columnnames_discretize.append(columnname)
            
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), columnnames_one_hot), ('discretize', KBinsDiscretizer(), columnnames_discretize), ('ord', OrdinalEncoder(), columnnames_binary)], remainder='passthrough', sparse_threshold=0)
    transformer.fit(X)        
        
    if return_df:
        return transformer
    else:
        return transformer    
    
    
def prepare_training(identifier, config):
    
    tf.random.set_seed(config['computation']['random_seed'])
    np.random.seed(config['computation']['random_seed'])
    random.seed(config['computation']['random_seed'])  
    
    config_test = deepcopy(config)
    #config_test['GRANDE']['steps'] = 100
    if 'REG' not in identifier:
        metrics = ['f1', 'roc_auc', 'accuracy']
        
        if 'BIN:' in identifier:
            config_test['GRANDE']['objective'] = 'classification'
            if 'loss' not in config_test['GRANDE']:
                config_test['GRANDE']['loss'] = 'crossentropy'     
        elif 'MULT:' in identifier:
            config_test['GRANDE']['objective'] = 'classification'    
            if 'loss' not in config_test['GRANDE']:
                config_test['GRANDE']['loss'] = 'kl_divergence'        
    else:
        metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        
        config_test['GRANDE']['objective'] = 'regression'
        if 'loss' not in config_test['GRANDE']:
            config_test['GRANDE']['loss'] = 'mse'   
        elif 'crossentropy' in config_test['GRANDE']['loss']:
            config_test['GRANDE']['loss'] = 'mse'     
                
    dataset_dict_list = get_preprocessed_dataset(identifier,
                                            random_seed=config_test['computation']['random_seed'],
                                            config=config_test,
                                            hpo=False,
                                            verbosity=0)    
    
    
    number_of_classes = len(np.unique(np.concatenate([dataset_dict_list[0]['y_train'].values, dataset_dict_list[0]['y_valid'].values, dataset_dict_list[0]['y_test'].values]))) if config_test['GRANDE']['objective'] == 'classification' else 1

    for i in range(config_test['computation']['cv_num_eval']):
        dataset_dict_list[i]['number_of_classes'] = number_of_classes
        dataset_dict_list[i]['number_of_variables'] = dataset_dict_list[0]['X_train'].shape[1]
    
    if identifier == 'BIN:Higgs':
        metrics.append('AMS')
    
    return dataset_dict_list, config_test, metrics

def get_jobs_by_gpu_memory(config):
    # Choose which GPU to use (if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get the amount of memory available on the device
    total_memory = torch.cuda.get_device_properties(device).total_memory

    # Convert bytes to gigabytes
    total_memory_mb = total_memory / (1024 ** 3) * 1_000
    
    gpu_vram = int(np.round(total_memory_mb))
    per_worker_memory = (config['GRANDE']['n_estimators'] * (2**config['GRANDE']['depth']+config['GRANDE']['depth']*config['GRANDE']['depth']) * config['GRANDE']['batch_size'])
    per_worker_memory_adjusted = np.round(per_worker_memory * 2 / (2**15))
    per_worker_memory_buffer = int(np.round(per_worker_memory_adjusted) * 1.5) + 1_000

    number_of_processes_by_memory = int(gpu_vram // per_worker_memory_buffer)
    
    print('Total GPU VRAM: {} MB'.format(gpu_vram))
    print('GPU VRAM per GRANDE Model with Buffer: {} MB'.format(per_worker_memory_buffer))
    print('Number of Processes per GPU: {}'.format(number_of_processes_by_memory))                                   
    
    return number_of_processes_by_memory


def load_hpo_results_by_dataset_and_timestring(dataset_name, timestring, identifier='test'):
    relevant_columnnames = []
    relevant_columnnames_rename = []
    try:
        hpo_results_summary = pd.read_csv("./hpo_results_summary/hpo_results_summary_" + dataset_name + ".csv")
    except FileNotFoundError:
        print('No file found for', dataset_name)
        return None
    for column in hpo_results_summary.columns:
        if timestring in column:
            if identifier + ' scores' in column:
                #relevant_columnnames.append(dataset_name + ' ' +column[:-(len(timestring)+len(identifier + ' scores')+1)])
                relevant_columnnames.append(column)
                relevant_columnnames_rename.append(dataset_name + ' ' +column[:-(len(timestring)+1)])
                
    if len(relevant_columnnames) > 0:
        hpo_results_summary_selected = hpo_results_summary[relevant_columnnames]
        hpo_results_summary_selected.columns = relevant_columnnames_rename
    else:
        print(dataset_name, 'not in results with timestring', timestring)
        return None        

    return hpo_results_summary_selected


def get_hpo_results_summary_by_dataset(dataset_identifier_list, timestring, model_identifier_list, identifier='test'):
    index_diff = 0
    dataset_identifier_list_reduced = []
    for i, dataset_name in enumerate(dataset_identifier_list):
        i = i-index_diff
        hpo_results_summary = load_hpo_results_by_dataset_and_timestring(dataset_name,  timestring, identifier=identifier)
        if hpo_results_summary is None:
            index_diff += 1
            continue
        if i == 0:
            hpo_results_summary_all_datasets = hpo_results_summary
        else:
            hpo_results_summary_all_datasets = pd.concat([hpo_results_summary_all_datasets, hpo_results_summary], axis=1)
        dataset_identifier_list_reduced.append(dataset_name)
    dataset_identifier_list = dataset_identifier_list_reduced
    
    relevant_columns = []
    for column in hpo_results_summary_all_datasets.columns:
        if 'GRANDE' in column:
            relevant_columns.append(column)  
    mean_column = hpo_results_summary_all_datasets[relevant_columns].mean(axis=1).values
    hpo_results_summary_all_datasets_mean = pd.DataFrame(data=mean_column, index=hpo_results_summary_all_datasets.index, columns=['GRANDE'])

    for i, model_identifier in enumerate(model_identifier_list):
        relevant_columns = []
        for column in hpo_results_summary_all_datasets.columns:
            if model_identifier in column:
                relevant_columns.append(column)    
        mean_column = hpo_results_summary_all_datasets[relevant_columns].mean(axis=1).values
        hpo_results_summary_all_datasets_mean_column = pd.DataFrame(data=mean_column, index=hpo_results_summary_all_datasets.index, columns=[model_identifier])
        hpo_results_summary_all_datasets_mean = pd.concat([hpo_results_summary_all_datasets_mean, hpo_results_summary_all_datasets_mean_column], axis=1)

    return hpo_results_summary_all_datasets_mean

def get_hpo_results_summary_by_dataset_normalized(dataset_identifier_list, timestring, model_identifier_list, identifier='test'):
    
    def normalize_no_variable(df, normalization_type="quantile", quantile=0.1):
        columns=df.columns
        df=df.values
        if normalization_type == "quantile":

            df_normalized = (df - np.quantile(df, q=0.1)) / (df.max() - np.quantile(df, q=0.1))

        elif normalization_type == "max":
            df_normalized = df / df.max()

        return pd.DataFrame(df_normalized, columns=columns)

    index_diff = 0
    dataset_identifier_list_reduced = []
    for i, dataset_name in enumerate(dataset_identifier_list):
        i = i-index_diff
        hpo_results_summary = load_hpo_results_by_dataset_and_timestring(dataset_name,  timestring, identifier=identifier)
        if hpo_results_summary is None:
            index_diff += 1
            continue
        if i == 0:
            hpo_results_summary_all_datasets = hpo_results_summary
        else:
            hpo_results_summary_all_datasets = pd.concat([hpo_results_summary_all_datasets, hpo_results_summary], axis=1)
        dataset_identifier_list_reduced.append(dataset_name)
    dataset_identifier_list = dataset_identifier_list_reduced
            
    hpo_results_summary_all_datasets = hpo_results_summary_all_datasets.dropna(axis=0, how='all')     
    for i, dataset_name in enumerate(dataset_identifier_list):
        relevant_columns = []
        for column in hpo_results_summary_all_datasets.columns:
            if dataset_name in column:
                relevant_columns.append(column)    
        hpo_results_summary_all_datasets[relevant_columns] = normalize_no_variable(hpo_results_summary_all_datasets[relevant_columns])
    hpo_results_summary_all_datasets[hpo_results_summary_all_datasets < 0] = 0

        
    relevant_columns = []
    for column in hpo_results_summary_all_datasets.columns:
        if 'GRANDE' in column:
            relevant_columns.append(column)  
    mean_column = hpo_results_summary_all_datasets[relevant_columns].mean(axis=1).values
    hpo_results_summary_all_datasets_mean = pd.DataFrame(data=mean_column, index=hpo_results_summary_all_datasets.index, columns=['GRANDE'])

    for i, model_identifier in enumerate(model_identifier_list):
        relevant_columns = []
        for column in hpo_results_summary_all_datasets.columns:
            if model_identifier in column:
                relevant_columns.append(column)    
        mean_column = hpo_results_summary_all_datasets[relevant_columns].mean(axis=1).values
        hpo_results_summary_all_datasets_mean_column = pd.DataFrame(data=mean_column, index=hpo_results_summary_all_datasets.index, columns=[model_identifier])
        hpo_results_summary_all_datasets_mean = pd.concat([hpo_results_summary_all_datasets_mean, hpo_results_summary_all_datasets_mean_column], axis=1)


    return hpo_results_summary_all_datasets_mean

def get_dataset_specs_by_identifier(identifier, config):
    config_dataset = deepcopy(config)
    config_dataset['computation']['cv_num_eval'] = 1
    config_dataset['computation']['cv_num_hpo'] = 1
    
    dataset_dict_list = get_preprocessed_dataset(identifier, 
                                             random_seed=42, 
                                             config=config_dataset,
                                             hpo=False,
                                             verbosity=0)
    dataset_dict = dataset_dict_list[0]
    
    X_data = pd.concat([dataset_dict['X_train'], dataset_dict['X_valid'], dataset_dict['X_test']])
    y_data = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid'], dataset_dict['y_test']])

    number_of_samples = X_data.shape[0]
    number_of_features = X_data.shape[1]
    number_of_classes = len(y_data.value_counts())
    
    return number_of_samples, number_of_features, number_of_classes

def sort_dataset_identifiers(dataset_identifier_list, config, by='features'): #features, samples, classes
    identifier_list_with_specs = []
    for identifier in dataset_identifier_list:
        number_of_samples, number_of_features, number_of_classes = get_dataset_specs_by_identifier(identifier, config)
        identifier_list_with_specs.append([identifier, number_of_samples, number_of_features, number_of_classes])
    
    if by=='features':
        identifier_list_with_specs_sorted = sorted(identifier_list_with_specs, key = lambda x: int(x[2]))
    if by=='samples':
        identifier_list_with_specs_sorted = sorted(identifier_list_with_specs, key = lambda x: int(x[1]))
    if by=='classes':
        identifier_list_with_specs_sorted = sorted(identifier_list_with_specs, key = lambda x: int(x[3]))   
        
    identifier_list_sorted = [sublist[0] for sublist in identifier_list_with_specs_sorted]
    
    return identifier_list_sorted

def get_optimizer_by_name(optimizer_name, learning_rate, warmup_steps, steps_per_epoch, cosine_decay_steps):
    
    if warmup_steps > 0:
        learning_rate = tfm.optimization.LinearWarmup(after_warmup_lr_sched=learning_rate,
                                                      warmup_learning_rate=learning_rate/warmup_steps,
                                                      warmup_steps=warmup_steps) 

    if cosine_decay_steps > 0:
        if cosine_decay_steps > 1:
            learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
                                                                                initial_learning_rate=learning_rate,
                                                                                first_decay_steps=cosine_decay_steps,
                                                                                #first_decay_steps=steps_per_epoch,
                                                                            )
        else:
            learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
                                                                                initial_learning_rate=learning_rate,
                                                                                first_decay_steps=max(10, int(steps_per_epoch*cosine_decay_steps)),
                                                                            )    
    if optimizer_name== 'QHM':
        optimizer = QHMOptimizer(learning_rate=1.0, nu=0.7, momentum=0.999)#QHMOptimizer(learning_rate=self.learning_rate_tree_index)
    elif optimizer_name== 'QHAdam':
        optimizer = QHAdamOptimizer(learning_rate=learning_rate, nu1=0.7, nu2=1.0, beta1=0.95, beta2=0.998)
        # nus=(0.7, 1.0), betas=(0.95, 0.998) #
    elif optimizer_name== 'AdaBelief':
        optimizer = tfa.optimizers.AdaBelief()
        optimizer.learning_rate = learning_rate
    elif optimizer_name== 'COCOB':
        optimizer = tfa.optimizers.COCOB(alpha=100)
    elif optimizer_name== 'Yogi':
        optimizer = tfa.optimizers.Yogi()
        optimizer.learning_rate = learning_rate
    elif optimizer_name== 'Lookahead':
        optimizer = tfa.optimizers.Lookahead(tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)) #beta_1=0.9, beta_2=0.999
    elif optimizer_name== 'NovoGrad':
        optimizer = tfa.optimizers.NovoGrad()
        optimizer = learning_rate
    elif optimizer_name== 'SWA':
        optimizer = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), average_period=5)
    elif optimizer_name== 'GradientAccumulator':
        #optimizer = GradientAccumulateOptimizer(accum_steps=min(steps_per_epoch, 1_000), optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))
        optimizer = GradientAccumulateOptimizer(accum_steps=steps_per_epoch, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))
    
    elif optimizer_name== 'MovingAverage':
        optimizer = tfa.optimizers.MovingAverage(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))
    elif optimizer_name== 'AdamW':
        optimizer = tf.keras.optimizers.AdamW()  
    elif optimizer_name== 'Lion':
        optimizer = tf.keras.optimizers.Lion(learning_rate=learning_rate, weight_decay=None)     
    elif optimizer_name== 'Adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.get(optimizer_name)
        optimizer.learning_rate = learning_rate
        
    #optimizer._distribution_strategy = None
        
    return optimizer


def poly1_cross_entropy(number_of_classes, epsilon=1.0, base_loss='crossentropy', focalLossGamma=2, no_logits=False, objective='classification', class_weights=False, weight_dict=None):
    def _poly1_cross_entropy(y_true, y_pred, sample_weight=None):

        # pt, CE, and Poly1 have shape [batch].    
        from_logits = not no_logits
        if base_loss == 'crossentropy':
            if number_of_classes == 2:
                loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits) #tf.keras.losses.get('binary_crossentropy')
            else:
                loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits) #tf.keras.losses.get('categorical_crossentropy')
        elif base_loss == 'focal_crossentropy':
            if class_weights:
                if number_of_classes == 2:
                    loss_function = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=focalLossGamma, apply_class_balancing=True, from_logits=from_logits)
                else:
                    loss_function = SparseCategoricalFocalLoss(gamma=focalLossGamma, class_weight=weight_dict['class_weights'], from_logits=from_logits)
            else:
                if number_of_classes == 2:
                    loss_function = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=focalLossGamma, apply_class_balancing=False, from_logits=from_logits)
                else:
                    loss_function = SparseCategoricalFocalLoss(gamma=focalLossGamma, from_logits=from_logits)                    
        else:
            loss_function = tf.keras.losses.get(base_loss) 
            try:
                 loss_function.from_logits = from_logits
            except:
                pass
            
        if objective == 'classification':
            if class_weights:
                if number_of_classes == 2:
                    if base_loss == 'focal_crossentropy':
                        if not no_logits:
                            loss_raw = tf.reduce_mean(loss_function(y_true, y_pred))
                        else:
                            loss_raw = tf.reduce_mean(loss_function(y_true, tf.math.sigmoid(y_pred)))                    
                    else:
                        if not no_logits:
                            loss_raw = tf.reduce_mean(loss_function(y_true, y_pred, sample_weight=sample_weight))
                        else:
                            loss_raw = tf.reduce_mean(loss_function(y_true, tf.math.sigmoid(y_pred), sample_weight=sample_weight))
                else: 
                    if base_loss == 'focal_crossentropy':
                        if not no_logits:
                            loss_raw = tf.reduce_mean(loss_function(y_true, y_pred))
                        else:
                            loss_raw = tf.reduce_mean(loss_function(y_true, tf.keras.activations.softmax(y_pred)))                    
                    else:
                        if not no_logits:
                            loss_raw = tf.reduce_mean(loss_function(y_true, y_pred, sample_weight=sample_weight))
                        else:
                            loss_raw = tf.reduce_mean(loss_function(y_true, tf.keras.activations.softmax(y_pred), sample_weight=sample_weight))

            else:
                if number_of_classes == 2:
                    if base_loss == 'focal_crossentropy':
                        if not no_logits:
                            loss_raw = tf.reduce_mean(loss_function(y_true, y_pred))
                        else:
                            loss_raw = tf.reduce_mean(loss_function(y_true, tf.math.sigmoid(y_pred)))                    
                    else:
                        if not no_logits:
                            loss_raw = tf.reduce_mean(loss_function(y_true, y_pred))
                        else:
                            loss_raw = tf.reduce_mean(loss_function(y_true, tf.math.sigmoid(y_pred)))
                else: 
                    if base_loss == 'focal_crossentropy':
                        if not no_logits:
                            loss_raw = tf.reduce_mean(loss_function(y_true, y_pred))
                        else:
                            loss_raw = tf.reduce_mean(loss_function(y_true, tf.keras.activations.softmax(y_pred)))                    
                    else:
                        if not no_logits:
                            loss_raw = tf.reduce_mean(loss_function(y_true, y_pred))
                        else:
                            loss_raw = tf.reduce_mean(loss_function(y_true, tf.keras.activations.softmax(y_pred)))
                  

        else:   
            loss_raw = tf.reduce_mean(loss_function(y_true, y_pred))              
        
        if number_of_classes > 2:
            if no_logits:
                pt = tf.reduce_sum(y_true * tf.nn.softmax(y_pred), axis=-1)
            else:
                pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        else:
            if no_logits:
                pt = tf.reduce_sum(tf.cast(tf.one_hot(tf.cast(tf.round(y_true), tf.int64), depth=number_of_classes), tf.float32) * tf.stack([1-y_pred, y_pred], axis=1), axis=-1)                
            else:
                pt = tf.reduce_sum(tf.cast(tf.one_hot(tf.cast(tf.round(y_true), tf.int64), depth=number_of_classes), tf.float32) * tf.stack([1-tf.math.sigmoid(y_pred), tf.math.sigmoid(y_pred)], axis=1), axis=-1)         



        Poly1 = loss_raw + epsilon * (1 - pt)
        loss = tf.reduce_mean(Poly1)
        return loss
    return _poly1_cross_entropy

                            
def pepare_GRANDE_for_training(config_training, dataset_dict, number_of_classes, timestr, cv=False):
    
    if cv:
        batch_size = min(config_training['GRANDE']['batch_size'], int(np.ceil(dataset_dict['X_train_cv'].shape[0]/2)))
        while dataset_dict['X_train_cv'].shape[0] % batch_size == 1:
            batch_size = batch_size + 1 
        if config_training['GRANDE']['objective'] == 'classification':
            class_weight = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(dataset_dict['y_train_cv']), y = dataset_dict['y_train_cv'])
            #class_weight = class_weight_raw/sum(class_weight)
            sample_weight_train = sklearn.utils.class_weight.compute_sample_weight(class_weight = 'balanced', y = dataset_dict['y_train_cv'])
            #sample_weight_train = sample_weight_train/sum(class_weight_raw)#(sample_weight_train-np.min(sample_weight_train))/(np.max(sample_weight_train)-np.min(sample_weight_train))
        else:
            class_weight = np.ones_like(np.unique(dataset_dict['y_train_cv']))
            sample_weight_train = np.ones_like(dataset_dict['y_train_cv'])
        sample_weight_train = tf.reshape(sample_weight_train, [-1, 1])      
    
    else:  
        batch_size = min(config_training['GRANDE']['batch_size'], int(np.ceil(dataset_dict['X_train'].shape[0]/2)))
        while dataset_dict['X_train'].shape[0] % batch_size == 1:
            batch_size = batch_size + 1 
        if config_training['GRANDE']['objective'] == 'classification':
            class_weight = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(dataset_dict['y_train']), y = dataset_dict['y_train'])
            #class_weight = class_weight_raw/sum(class_weight)
            sample_weight_train = sklearn.utils.class_weight.compute_sample_weight(class_weight = 'balanced', y = dataset_dict['y_train'])
            #sample_weight_train = sample_weight_train/sum(class_weight_raw)#(sample_weight_train-np.min(sample_weight_train))/(np.max(sample_weight_train)-np.min(sample_weight_train))
        else:
            class_weight = np.ones_like(np.unique(dataset_dict['y_train']))
            sample_weight_train = np.ones_like(dataset_dict['y_train'])
        sample_weight_train = tf.reshape(sample_weight_train, [-1, 1])   


    from_logits = not config_training['GRANDE']['logit_weights']
    if config_training['GRANDE']['polyLoss']:
        #print('PolyLoss Currently not implemented')
        loss_function = poly1_cross_entropy(number_of_classes, 
                                            epsilon=config_training['GRANDE']['polyLossEpsilon'], 
                                            base_loss=config_training['GRANDE']['loss'], 
                                            focalLossGamma=config_training['GRANDE']['focalLossGamma'], 
                                            no_logits=config_training['GRANDE']['logit_weights'], 
                                            objective=config_training['GRANDE']['objective'], 
                                            class_weights=config_training['GRANDE']['class_weights'],
                                            weight_dict={'class_weights': class_weight})

    else:
        if config_training['GRANDE']['loss'] == 'crossentropy':

            if number_of_classes == 2:
                loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits) #tf.keras.losses.get('binary_crossentropy')
            else:
                loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits) #tf.keras.losses.get('categorical_crossentropy')
        elif config_training['GRANDE']['loss'] == 'focal_crossentropy':
            if config_training['GRANDE']['class_weights']:
                if number_of_classes == 2:
                    loss_function = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=config_training['GRANDE']['focalLossGamma'], apply_class_balancing=True, from_logits=from_logits)
                else:
                    loss_function = SparseCategoricalFocalLoss(gamma=config_training['GRANDE']['focalLossGamma'], class_weight=class_weight, from_logits=from_logits)
            else:
                if number_of_classes == 2:
                    loss_function = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=config_training['GRANDE']['focalLossGamma'], apply_class_balancing=False, from_logits=from_logits)
                else:
                    loss_function = SparseCategoricalFocalLoss(gamma=config_training['GRANDE']['focalLossGamma'], from_logits=from_logits)                    
        else:
            if config_training['GRANDE']['objective'] == 'regression':
                if cv:
                    loss_function = loss_function_regression(loss_name=config_training['GRANDE']['loss'], mean=np.mean(dataset_dict['y_train_cv']), std=np.std(dataset_dict['y_train_cv']), transformation_type=config_training['GRANDE']['transformation_type'])
                else:
                    loss_function = loss_function_regression(loss_name=config_training['GRANDE']['loss'], mean=np.mean(dataset_dict['y_train']), std=np.std(dataset_dict['y_train']), transformation_type=config_training['GRANDE']['transformation_type'])
            else:
                loss_function = tf.keras.losses.get(config_training['GRANDE']['loss'])  
                try:
                     loss_function.from_logits = from_logits
                except:
                    pass                

    if cv:
        train_data = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(tf.convert_to_tensor(dataset_dict['X_train_cv']), tf.float32),
                                                     tf.dtypes.cast(tf.convert_to_tensor(dataset_dict['y_train_cv']), tf.float32)))    
    else:
        train_data = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(tf.convert_to_tensor(dataset_dict['X_train']), tf.float32),
                                                     tf.dtypes.cast(tf.convert_to_tensor(dataset_dict['y_train']), tf.float32)))
    if config_training['GRANDE']['data_subset_fraction'] < 1.0:
        train_data = (train_data
                #.shuffle(32_768)
                .cache()
                #.repeat()  
                .batch(batch_size=batch_size, drop_remainder=True) #, drop_remainder=True, num_parallel_calls=None
                .prefetch(AUTOTUNE)      #AUTOTUNE
                     )    
    else:
        train_data = (train_data
                .shuffle(32_768)
                .cache()
                #.repeat()  
                .batch(batch_size=batch_size, drop_remainder=config_training['GRANDE']['drop_remainder']) #, drop_remainder=True, num_parallel_calls=None
                .prefetch(AUTOTUNE)      #AUTOTUNE
                     )

    if cv:
        batch_size_val = min(config_training['GRANDE']['batch_size'], dataset_dict['X_valid_cv'].shape[0])
        #batch_size_val = min(config_training['GRANDE']['batch_size'], int(np.ceil(dataset_dict['X_valid_cv'].shape[0]/2)))
        while dataset_dict['X_valid_cv'].shape[0] % batch_size_val == 1:
            batch_size_val = batch_size_val + 1 
        if config_training['GRANDE']['objective'] == 'classification':
            if cv:
                class_weight_dict = dict(map(lambda i,j : (i,j) , np.unique(dataset_dict['y_train_cv']), class_weight))
            else:
                class_weight_dict = dict(map(lambda i,j : (i,j) , np.unique(dataset_dict['y_train']), class_weight))
            sample_weight_valid = sklearn.utils.class_weight.compute_sample_weight(class_weight = class_weight_dict, y = dataset_dict['y_valid_cv'])
            #sample_weight_valid = sample_weight_valid/sum(class_weight_raw)#(sample_weight_valid-np.min(sample_weight_valid))/(np.max(sample_weight_valid)-np.min(sample_weight_valid))
        else:
            sample_weight_valid = np.ones_like(dataset_dict['y_valid_cv'])
        sample_weight_valid = tf.reshape(sample_weight_valid, [-1, 1])

        valid_data = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(tf.convert_to_tensor(dataset_dict['X_valid_cv']), tf.float32), 
                                                         tf.dtypes.cast(tf.convert_to_tensor(dataset_dict['y_valid_cv']), tf.float32)))

    
    else:
        
        batch_size_val = min(config_training['GRANDE']['batch_size'], dataset_dict['X_valid'].shape[0])
        #batch_size_val = min(config_training['GRANDE']['batch_size'], int(np.ceil(dataset_dict['X_valid'].shape[0]/2)))
        while dataset_dict['X_valid'].shape[0] % batch_size_val == 1:
            batch_size_val = batch_size_val + 1 
        if config_training['GRANDE']['objective'] == 'classification':
            if cv:
                class_weight_dict = dict(map(lambda i,j : (i,j) , np.unique(dataset_dict['y_train_cv']), class_weight))
            else:
                class_weight_dict = dict(map(lambda i,j : (i,j) , np.unique(dataset_dict['y_train']), class_weight))
            sample_weight_valid = sklearn.utils.class_weight.compute_sample_weight(class_weight = class_weight_dict, y = dataset_dict['y_valid'])
            #sample_weight_valid = sample_weight_valid/sum(class_weight_raw)#(sample_weight_valid-np.min(sample_weight_valid))/(np.max(sample_weight_valid)-np.min(sample_weight_valid))
        else:
            sample_weight_valid = np.ones_like(dataset_dict['y_valid'])
        sample_weight_valid = tf.reshape(sample_weight_valid, [-1, 1])

        valid_data = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(tf.convert_to_tensor(dataset_dict['X_valid']), tf.float32), 
                                                         tf.dtypes.cast(tf.convert_to_tensor(dataset_dict['y_valid']), tf.float32)))


    valid_data = (valid_data
            #.shuffle(32_768)
            .cache()
            #.repeat()  
            .batch(batch_size=batch_size_val, drop_remainder=False) #, drop_remainder=True, num_parallel_calls=None
            .prefetch(AUTOTUNE)      
                 )              



    if config_training['GRANDE']['restart_type'] == 'metric' or config_training['GRANDE']['early_stopping_type'] == 'metric': 
        if config_training['GRANDE']['objective'] == 'classification':
            metrics_GRANDE = [F1ScoreSparse(average='macro', num_classes=number_of_classes, threshold=0.5)]
        else:
            if cv:
                metrics_GRANDE = [R2ScoreTransform(transformation_type=config_training['GRANDE']['transformation_type'], mean=np.mean(dataset_dict['y_train_cv']), std=np.std(dataset_dict['y_train_cv']))]
            
            else:
                metrics_GRANDE = [R2ScoreTransform(transformation_type=config_training['GRANDE']['transformation_type'], mean=np.mean(dataset_dict['y_train']), std=np.std(dataset_dict['y_train']))]
            #metrics_GRANDE = [tfa.metrics.RSquare()]
    else:
        metrics_GRANDE = []     

    if config_training['GRANDE']['early_stopping_type'] == 'metric':
        monitor = 'val_' + metrics_GRANDE[0].name
    else:
        monitor = 'val_loss'              

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                      patience=config_training['GRANDE']['early_stopping_epochs'], 
                                                      restore_best_weights=True)
    callbacks = [early_stopping]

    if config_training['GRANDE']['reduce_lr']:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=config_training['GRANDE']['reduce_lr_fraction'], patience=config_training['GRANDE']['early_stopping_epochs']//3)
        callbacks.append(reduce_lr)

    if cv:
        steps_per_epoch=dataset_dict['X_train_cv'].shape[0]//batch_size
    else:
        steps_per_epoch=dataset_dict['X_train'].shape[0]//batch_size

    optimizer_function_dict = {
        'weights_optimizer': get_optimizer_by_name(config_training['GRANDE']['optimizer'], config_training['GRANDE']['learning_rate_weights'], config_training['GRANDE']['warmup_steps'], steps_per_epoch=steps_per_epoch, cosine_decay_steps=config_training['GRANDE']['cosine_decay_steps']),
        'index_optimizer': get_optimizer_by_name(config_training['GRANDE']['optimizer'], config_training['GRANDE']['learning_rate_index'], config_training['GRANDE']['warmup_steps'], steps_per_epoch=steps_per_epoch, cosine_decay_steps=config_training['GRANDE']['cosine_decay_steps']),
        'values_optimizer': get_optimizer_by_name(config_training['GRANDE']['optimizer'], config_training['GRANDE']['learning_rate_values'], config_training['GRANDE']['warmup_steps'], steps_per_epoch=steps_per_epoch, cosine_decay_steps=config_training['GRANDE']['cosine_decay_steps']),
        'leaf_optimizer': get_optimizer_by_name(config_training['GRANDE']['optimizer'], config_training['GRANDE']['learning_rate_leaf'], config_training['GRANDE']['warmup_steps'], steps_per_epoch=steps_per_epoch, cosine_decay_steps=config_training['GRANDE']['cosine_decay_steps']),  
                              }
            


    if config_training['GRANDE']['class_weights'] and config_training['GRANDE']['objective']=='classification':
        if cv:
            class_weights = calculate_class_weights(dataset_dict['y_train_cv'])
        else:
            class_weights = calculate_class_weights(dataset_dict['y_train'])
        class_weight_dict = {}
        for i in range(number_of_classes):
            class_weight_dict[i] = class_weights[i]
    else:
        class_weight_dict = None    

    return (train_data, 
            valid_data, 
            batch_size, 
            batch_size_val, 
            class_weight_dict, 
            loss_function, 
            optimizer_function_dict, 
            metrics_GRANDE, callbacks)

  
class MyDictObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
            

            

class DecisionTree:
    def __init__(self, split_thresholds, one_hot_encodings, class_probabilities, leaf_weights, estimator_features):
        self.split_thresholds = split_thresholds
        self.one_hot_encodings = one_hot_encodings
        self.class_probabilities = class_probabilities
        self.leaf_weights = leaf_weights
        self.estimator_features = estimator_features
        self.max_depth = int(np.log2(len(class_probabilities)))
        self.root = self.build_tree()

    def build_tree(self, node_index=0):
        # If node index is greater than the number of internal nodes, we have reached a leaf node
        if node_index >= len(self.split_thresholds):
            return {'class': self.class_probabilities[node_index - len(self.split_thresholds)], 
                    'weight': self.leaf_weights[node_index - len(self.split_thresholds)]}

        # Otherwise, create a new node with the corresponding threshold value and recursively build its children
        threshold_values = self.split_thresholds[node_index]
        one_hot_encoding = self.one_hot_encodings[node_index]
        node = {'threshold_values': threshold_values, 'one_hot_encoding': one_hot_encoding}
        node['left'] = self.build_tree(2 * node_index + 1)
        node['right'] = self.build_tree(2 * node_index + 2)

        return node

    def prune_tree(self, data, min_samples):
        if min_samples < 1: #int(min_samples) - min_samples != 0: #if float number
            min_samples = max(1, data.shape[0] * min_samples)
        
        data_complete = deepcopy(data)
        
        self._pass_node(self.root, data)
        
        self.root_unpruned = deepcopy(self.root)
        self._prune_node(self.root, data, data_complete, min_samples)

    def _pass_node(self, node, data):
        
        node['num_samples_passed'] = data.shape[0]
        #print(node)
        if 'class' in node:
            return

        one_hot_encoding_argmax = np.argmax(node['one_hot_encoding'])
        feature_index = self.estimator_features[one_hot_encoding_argmax]
        
        left_indices = np.where(data[:, feature_index] <= node['threshold_values'][one_hot_encoding_argmax])[0]
        right_indices = np.where(data[:, feature_index] > node['threshold_values'][one_hot_encoding_argmax])[0]
                 
        #print('left_indices_test', np.where(data[:, np.argmax(node['one_hot_encoding'])] <= node['threshold_values'][np.argmax(node['one_hot_encoding'])]))
        #print('left_indices', left_indices)
        
            
        self._pass_node(node['left'], data[left_indices])
        self._pass_node(node['right'], data[right_indices])        
        
        
    def _prune_node(self, node, data, data_complete, min_samples):

        # If the node is a leaf, return
        if 'class' in node:
            return

        # Recursively prune the children of the node
        self._prune_node(node['left'], data, data_complete, min_samples)
        self._prune_node(node['right'], data, data_complete, min_samples)

        # Prune the node if the number of samples passing through it is less than the minimum
        
        samples_left = node['left']['num_samples_passed']
        samples_right = node['right']['num_samples_passed']
        
        #print('node', node)
        #print('node[left]', node['left'])
        
        if samples_left < min_samples and samples_right < min_samples:
            #print(node)
            if 'class' in node['left'] and 'class' in node['right']:
                #print(node['left']['class'], node['right']['class'])
                node['class'] = np.mean([node['left']['class'], node['right']['class']], axis=0)
                node['weight'] = np.mean([node['left']['weight'], node['right']['weight']], axis=0)
                #print('node 1', node['class'])
                node['num_samples_passed'] = np.sum([node['left']['num_samples_passed'], node['right']['num_samples_passed']])
                node.pop('left', None)
                node.pop('right', None) 
            else:
                print('SHOULD NOT HAPPEN, CHECK PLEASE')
                return       

        else:
            if samples_left < min_samples:
                #print('node[left]', node['left'])
                #print('node[right]', node['right'])   
                if 'class' in node['right']:                          
                    node['weight'] = node['right']['weight']              
                    node['class'] = node['right']['class']
                    
                    #print('node 2', node['class'])
                    node['num_samples_passed'] = node['right']['num_samples_passed']
                    node.pop('left', None)
                    node.pop('right', None) 
                else:
                    new_node = deepcopy(node['right'])
                    node['left'] = new_node['left']
                    node['right'] = new_node['right']
                    node['one_hot_encoding'] = new_node['one_hot_encoding']
                    node['threshold_values'] = new_node['threshold_values']                
            elif samples_right < min_samples:
                #print('node[left]', node['left'])
                #print('node[right]', node['right'])
                if 'class' in node['left']:               
                    node['weight'] = node['left']['weight']
                    node['class'] = node['left']['class']
                    #print('node 3', node['class'])
                    node['num_samples_passed'] = node['left']['num_samples_passed']
                    node.pop('left', None)
                    node.pop('right', None)
                else:
                    new_node = deepcopy(node['left'])
                    node['left'] = new_node['left']
                    node['right'] = new_node['right']
                    node['one_hot_encoding'] = new_node['one_hot_encoding']
                    node['threshold_values'] = new_node['threshold_values']
            else:
                return
        self._pass_node(self.root, data_complete)
                
    def predict(self, instance, node=None):
        # If no starting node is specified, start at the root of the tree
        if node is None:
            node = self.root

        # If we have reached a leaf node, return the corresponding class probability
        if 'class' in node:
            return node['class']

        # Otherwise, compare the instance's feature values to the node's threshold values and traverse the appropriate child
        threshold_values = node['threshold_values']
        one_hot_encoding = node['one_hot_encoding']
        feature_values = instance[one_hot_encoding == 1]
        if np.all(feature_values <= threshold_values):
            return self.predict(instance, node['left'])
        else:
            return self.predict(instance, node['right'])

    def evaluate(self, test_data, true_labels):
        num_correct = 0
        for i in range(len(test_data)):
            prediction = self.predict(test_data[i])
            if prediction == true_labels[i]:
                num_correct += 1
        accuracy = num_correct / len(test_data)
        return accuracy
    
    def extend_to_fully_grown(self):
        self.split_thresholds_unpruned = deepcopy(self.split_thresholds)
        self.one_hot_encodings_unpruned = deepcopy(self.one_hot_encodings)
        self.class_probabilities_unpruned = deepcopy(self.class_probabilities)
        self.root_pruned_extended = deepcopy(self.root)
        
        current_node_list = [self.root_pruned_extended]
        for current_depth in range(self.max_depth):
            current_node_list_new = []
            for current_node in current_node_list:
                if 'class' in current_node:
                    current_node_copy = deepcopy(current_node)
                    current_node.pop('class', None)
                    current_node.pop('weight', None)
                    current_node['threshold_values'] = np.zeros_like(self.split_thresholds_unpruned[0])
                    current_node['one_hot_encoding'] = np.zeros_like(self.one_hot_encodings_unpruned[0])
                    current_node['left'] = current_node_copy
                    current_node['right'] = current_node_copy
                    
                current_node_list_new.append(current_node['left'])
                current_node_list_new.append(current_node['right'])
            current_node_list = current_node_list_new
            
        self.to_array_representation(root_type='pruned')
            

    def plot_tree_from_array(self, filename='./tree_tmp.png', plot_format='png'):
        dot = graphviz.Digraph()
        dot.node('0', 'Root')
        self._plot_subtree_from_array(dot, 0)
        dot.render(filename, format=plot_format, view=True)

    def _plot_subtree_from_array(self, dot, node_index):
        if node_index >= len(self.split_thresholds):
            node_label = f'C: {self.class_probabilities[node_index - len(self.split_thresholds)]}'
        else:
            one_hot_encoding_argmax = self.one_hot_encodings[node_index].argmax()
            feature_index = self.estimator_features[one_hot_encoding_argmax]         
            
            node_label = f'F {feature_index}: <= {self.split_thresholds[node_index]}'
            left_child_index = 2 * node_index + 1
            right_child_index = 2 * node_index + 2
            dot.node(str(left_child_index), '')
            dot.node(str(right_child_index), '')
            dot.edge(str(node_index), str(left_child_index), 'True')
            dot.edge(str(node_index), str(right_child_index), 'False')
            self._plot_subtree_from_array(dot, left_child_index)
            self._plot_subtree_from_array(dot, right_child_index)
        dot.node(str(node_index), node_label)
        
    def plot_tree(self, filename='./tree_tmp', plot_format='png', root_type='current'): #initial, pruned_extended
        import graphviz
        dot = graphviz.Digraph()
        if root_type == 'current':
            self._plot_subtree(dot, self.root)
        elif root_type == 'initial':
            self._plot_subtree(dot, self.root_unpruned)
        elif root_type == 'pruned_extended':
            self._plot_subtree(dot, self.root_pruned_extended)
        else:
            print('Root type ' + root_type + ' not existing, taking current root')
            self._plot_subtree(dot, self.root)
            
        dot.render(filename, format=plot_format, view=False)
        display(dot)
        #dot.render(filename, view=True)

    def _plot_subtree(self, dot, node):
        if 'class' in node:
            class_value = tf.math.sigmoid(node["class"]).numpy()
            weight = float(node["weight"])
            try:
                num_samples_passed = node["num_samples_passed"]
            except:
                num_samples_passed = 0.0
            #node_label = f'Class: {class_value:.3f} Num Samples: {num_samples_passed:.0f}'
            node_label = f'P: {class_value:.3f} \n W: {weight:.3f} \n N: {num_samples_passed:.0f}'
        else:
            one_hot_encoding_argmax = node['one_hot_encoding'].argmax()
            feature_index = self.estimator_features[one_hot_encoding_argmax]             
            threshold_value = node["threshold_values"][one_hot_encoding_argmax]
            try:
                num_samples_passed = node["num_samples_passed"]
            except:
                num_samples_passed = 0            
            node_label = f'F {feature_index}: <= {threshold_value:.3f} \n N: {num_samples_passed:.0f}'
            left_child = node['left']
            right_child = node['right']
            dot.node(str(id(left_child)), '')
            dot.node(str(id(right_child)), '')
            dot.edge(str(id(node)), str(id(left_child)), 'True')
            dot.edge(str(id(node)), str(id(right_child)), 'False')
            self._plot_subtree(dot, left_child)
            self._plot_subtree(dot, right_child)
        dot.node(str(id(node)), node_label)
        

    def count_nodes(self, node=None):
        if node is None:
            node = self.root
            if 'class' in node:
                leaf = 1
            else:
                internal = 1
        
        if 'class' in node:
            return 0, 1#1, 0

        left_internal, left_leaf = self.count_nodes(node['left'])
        right_internal, right_leaf = self.count_nodes(node['right'])
        internal = left_internal + right_internal
        leaf = left_leaf + right_leaf
        if 'left' in node or 'right' in node:
            internal += 1

        return internal, leaf

def plot_tree_by_index(ensemble, tree_index, data_prune=None):
    
    split_values = ensemble.output_layer.split_values[tree_index].numpy()
    leaf_classes_array = ensemble.output_layer.leaf_classes_array[tree_index].numpy()#tf.sigmoid(self.leaf_classes_array).numpy()
    split_index_array = tf.cast(tfa.seq2seq.hardmax(ensemble.output_layer.split_index_array[tree_index]), tf.int64).numpy()
    leaf_weights = ensemble.output_layer.estimator_weights[tree_index].numpy()
    estimator_features = ensemble.output_layer.features_by_estimator[tree_index].numpy()
    
    tree = DecisionTree(split_values, split_index_array, leaf_classes_array, leaf_weights, estimator_features)

    if data_prune is not None:
        data_prune = enforce_numpy(data_prune)
        tree.prune_tree(data=data_prune, min_samples=1)          
        
        num_internal_nodes_acutal, num_leaf_nodes_acutal = tree.count_nodes()
    else:
        num_internal_nodes_acutal = ensemble.output_layer.split_values.shape[1]
        num_leaf_nodes_acutal = ensemble.output_layer.split_index_array.shape[1]
        

    print('Internal Nodes Pruned:', num_internal_nodes_acutal)
    print('Leaf Nodes Pruned:', num_leaf_nodes_acutal)
    
    tree.plot_tree()



def top_k_elements_per_row(matrix, k):
    """
    Get top k elements for each row of a matrix.

    Parameters:
    - matrix: 2D NumPy array
    - k: number of top elements to retrieve per row

    Returns:
    - 2D array with top k elements per row, in descending order
    """
    # Get the indices that would sort each row
    sorted_indices = np.argsort(matrix, axis=1)
    
    # Get the top k indices for each row
    top_k_indices = sorted_indices[:, -k:]
    
    # Use numpy advanced indexing to get the elements
    # Rows range from 0 to number of rows
    # Columns use the top_k_indices
    rows = np.arange(matrix.shape[0])[:, None]
    top_k_elements = matrix[rows, top_k_indices]
    
    # Reverse the order of elements for each row to get them in descending order
    return np.flip(top_k_elements, axis=1)

def top_k_indices_per_row(matrix, k):
    """
    Get top k indices for each row of a matrix.

    Parameters:
    - matrix: 2D NumPy array
    - k: number of top indices to retrieve per row

    Returns:
    - 2D array with top k indices per row, corresponding to values in descending order
    """
    # Get the indices that would sort each row
    sorted_indices = np.argsort(matrix, axis=1)
    
    # Get the top k indices for each row
    top_k_indices = sorted_indices[:, -k:]
    
    # Reverse the order of indices for each row to get them corresponding to values in descending order
    return np.flip(top_k_indices, axis=1)

def pairwise_concatenate(A, B):
    """
    Concatenate two 2D matrices pair-wise along axis 1.

    Parameters:
    - A, B: 2D NumPy arrays of the same shape

    Returns:
    - 2D array with concatenated columns from A and B
    """
    # Reshape the matrices to introduce a new axis
    A_reshaped = A[:, :, np.newaxis]
    B_reshaped = B[:, :, np.newaxis]
    
    # Concatenate the matrices along the new axis
    concatenated = np.concatenate((A_reshaped, B_reshaped), axis=2)
    
    # Reshape back to 2D
    result = concatenated.reshape(A.shape[0], -1)
    
    return result


def inverse_sigmoid(y):
    return np.log(y / (1.0 - y))

def get_tree_paths(tree, instance, feature_names, path=[]):
    """
    Function to get the path of an instance through a decision tree using the tree structure.
    """
    # If it's a leaf node, return the path
    if 'leaf' in tree:
        return [(path, tree['leaf'])]
    
    # Extract the split feature and threshold
    feature_index = feature_names.index(tree['split'])#int(tree['split'])
    threshold = tree['split_condition']
    
    # Determine the child node to traverse to
    if instance[feature_index] < threshold:
        next_node = [child for child in tree['children'] if child['nodeid'] == tree['yes']][0]
        return get_tree_paths(next_node, instance, feature_names, path + [f"{feature_names[feature_index]} < {threshold:.2f}"])
    else:
        next_node = [child for child in tree['children'] if child['nodeid'] == tree['no']][0]
        return get_tree_paths(next_node, instance, feature_names, path + [f"{feature_names[feature_index]} >= {threshold:.2f}"])

def scale_absolute_to_sum_one(values):
    total_absolute = sum(abs(v) for v in values)
    if total_absolute == 0:  # Handle the case where the sum of absolute values is 0 to avoid division by zero
        return [1.0 / len(values) for _ in values]  # Distribute equally among all values
    
    scaled_values = [v / total_absolute for v in values]
    return scaled_values
