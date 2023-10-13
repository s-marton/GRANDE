import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from typing import Callable

from livelossplot import PlotLosses

import os
import gc
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from IPython.display import Image
from IPython.display import display, clear_output

import pandas as pd
import sys

import warnings
warnings.filterwarnings('ignore')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
#os.environ["PYTHONWARNINGS"] = "default"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["PYTHONWARNINGS"] = "ignore"

import logging

import tensorflow as tf
import tensorflow_addons as tfa

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
#tf.get_logger().setLevel('WARNING')
#tf.autograph.set_verbosity(1)

np.seterr(all="ignore")

from keras import backend as K

import seaborn as sns
sns.set_style("darkgrid")

import time
import random

from utilities.utilities_GRANDE import *
from utilities.GRANDE import *
#from utilities.DNDT import *

from gradient_accumulator import GradientAccumulateOptimizer

from joblib import Parallel, delayed
from gpuparallel import GPUParallel, delayed


from itertools import product
from collections.abc import Iterable

from copy import deepcopy

from tensorflow.data import AUTOTUNE
#from tensorflow.data import AUTOTUNE

from qhoptim.tf import QHMOptimizer, QHAdamOptimizer
from focal_loss import SparseCategoricalFocalLoss

     
class GRANDE(tf.keras.Model):
    def __init__(self, 
                 #number_of_classes,
                 #number_of_variables,
                 #objective,
                 **kwargs):
        
        super(GRANDE, self).__init__()
        
        if 'verbosity' not in kwargs.keys():
            kwargs['verbosity'] = 0 
            
        self.config = kwargs
        for arg_key, arg_value in kwargs.items():
            if arg_key == 'optimizer':
                setattr(self, 'optimizer_name', arg_value)                 
            elif arg_key == 'loss':
                setattr(self, 'loss_name', arg_value)                 
            else:
                setattr(self, arg_key, arg_value)     
                
        self.leaf_trainable = True
        self.split_index_trainable = True
        self.split_values_trainable = True
        self.weights_trainable = True
        
        tf.keras.utils.set_random_seed(self.random_seed + self.model_seed)

        config_block = self.config
        self.output_layer = GRANDEBlock(**config_block)                        

    @tf.function(jit_compile=True)
    def call(self, inputs, training):
        output = self.output_layer(inputs)

        if self.output_layer.data_subset_fraction < 1.0 and training:
            result = tf.scatter_nd(indices=tf.expand_dims(self.output_layer.data_select, 2), updates=tf.transpose(output), shape=[tf.shape(inputs)[0]])
            result = (result / self.output_layer.counts) * self.output_layer.n_estimators   
        else:
            if self.objective == 'regression' or self.number_of_classes == 2:                                   
                result = tf.einsum('ij->i', output)
            else:                    
                result = tf.einsum('ijl->ij', output)

        if self.objective == 'regression' or self.number_of_classes == 2: 
            result = tf.expand_dims(result, 1)

        return result

    @tf.function(jit_compile=True)
    def output_with_weights(self, inputs, training=False):

        output, estimator_weights_leaf_softmax = self.output_layer.output_with_weights(inputs)

        if self.output_layer.data_subset_fraction < 1.0 and training:
            result = tf.scatter_nd(indices=tf.expand_dims(self.output_layer.data_select, 2), updates=tf.transpose(output), shape=[tf.shape(inputs)[0]])
            result = (result / self.output_layer.counts) * self.output_layer.n_estimators   
        else:
            if self.objective == 'regression' or self.number_of_classes == 2:                                   
                result = tf.einsum('ij->i', output)
            else:                    
                result = tf.einsum('ijl->ij', output)
        
        if self.objective == 'regression' or self.number_of_classes == 2: 
            result = tf.expand_dims(result, 1)

        return result, estimator_weights_leaf_softmax
    
        
    def train_step(self, data):
    
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        if not self.built:
            self(x) 

        with tf.GradientTape() as weights_tape:
            with tf.GradientTape() as index_tape:
                with tf.GradientTape() as values_tape:
                    with tf.GradientTape() as leaf_tape:                        
                        weights_tape.watch(self.output_layer.estimator_weights)          
                        index_tape.watch(self.output_layer.split_index_array)
                        values_tape.watch(self.output_layer.split_values)
                        leaf_tape.watch(self.output_layer.leaf_classes_array)
                        
                        y_pred = self(x, training=True) 

                        # Compute the loss
                        loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
        if self.weights_trainable:
            weights_gradients = weights_tape.gradient(loss, [self.output_layer.estimator_weights])
            self.weights_optimizer.apply_gradients(zip(weights_gradients, [self.output_layer.estimator_weights]))       

        if self.split_index_trainable:
            index_gradients = index_tape.gradient(loss, [self.output_layer.split_index_array])
            self.index_optimizer.apply_gradients(zip(index_gradients, [self.output_layer.split_index_array]))
             
        if self.split_values_trainable:
            values_gradients = values_tape.gradient(loss, [self.output_layer.split_values])
            self.values_optimizer.apply_gradients(zip(values_gradients, [self.output_layer.split_values]))
            
        if self.leaf_trainable:
            leaf_gradients = leaf_tape.gradient(loss, [self.output_layer.leaf_classes_array])  
            self.leaf_optimizer.apply_gradients(zip(leaf_gradients, [self.output_layer.leaf_classes_array]))        

        # Update metrics (optional)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current values
        return {m.name: m.result() for m in self.metrics}
    
    
               
    
    def predict(self, X, batch_size, verbose):
        preds = super(GRANDE, self).predict(X, batch_size, verbose=0)
        #####tf.print('preds raw', preds)         
        if self.objective == 'regression':
            if self.transformation_type == 'mean':
                preds = preds * self.std + self.mean
            elif self.transformation_type == 'log':
                preds = tf.exp(preds) - 1e-7    
        else:
            if not self.logit_weights:
                if self.number_of_classes <= 2:
                    preds = tf.math.sigmoid(preds) 
                else:
                    preds = tf.keras.activations.softmax(preds) 
        #####tf.print('preds final', preds)         
        return preds    
    

    def set_params(self, **kwargs): 
                
        for arg_key, arg_value in kwargs.items():
            if arg_key == 'optimizer':
                setattr(self, 'optimizer_name', arg_value)                 
            elif arg_key == 'loss':
                setattr(self, 'loss_name', arg_value)                 
            else:
                setattr(self, arg_key, arg_value)                 
        
        tf.keras.utils.set_random_seed(self.random_seed + self.model_seed)

        config_block = self.config
        self.output_layer = GRANDEBlock(**config_block)       
                 
                
    def get_params(self):
        return self.config    
    
    
class GRANDEBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 name='GRANDEBlock',
                 trainable=True,
                 dtype='float',
                 intermediate=False,
                 **kwargs):
        super(GRANDEBlock, self).__init__()

        for arg_key, arg_value in kwargs.items():
            if arg_key == 'optimizer':
                setattr(self, 'optimizer_name', arg_value)                 
            elif arg_key == 'loss':
                setattr(self, 'loss_name', arg_value)                 
            else:
                setattr(self, arg_key, arg_value)  
                
        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth
        self.intermediate = intermediate

        tf.keras.utils.set_random_seed(self.random_seed + self.model_seed)
  
        
    def build(self, input_shape):

        tf.keras.utils.set_random_seed(self.random_seed + self.model_seed)

        self.data_shape = self.number_of_variables
        self.number_of_variables = input_shape[-1]   
        
        if self.selected_variables > 1:
            self.selected_variables = min(self.selected_variables, self.number_of_variables)
        else:
            self.selected_variables = int(self.number_of_variables * self.selected_variables)
            self.selected_variables = min(self.selected_variables, 50)
            self.selected_variables = max(self.selected_variables, 10)
            self.selected_variables = min(self.selected_variables, self.number_of_variables)  

        if self.data_subset_fraction < 1.0:
            self.subset_size = tf.cast(self.batch_size * self.data_subset_fraction, tf.int32)
            if self.bootstrap:
                self.data_select = tf.random.uniform(shape=(self.n_estimators, self.subset_size), minval=0, maxval=self.batch_size, dtype=tf.int32)
            else:
                    
                indices = [np.random.choice(self.batch_size, size=self.subset_size, replace=False) for _ in range(self.n_estimators)]
                self.data_select = tf.stack(indices)

            items, self.counts = np.unique(self.data_select, return_counts=True)
            self.counts = tf.constant(self.counts, dtype=tf.float32)        
        
        if self.n_estimators > 1:
            self.features_by_estimator = tf.stack([np.random.choice(self.number_of_variables, size=(self.selected_variables), replace=False, p=None) for _ in range(self.n_estimators)])  
        else:
            self.features_by_estimator = tf.constant([np.random.choice(self.number_of_variables, size=(self.selected_variables), replace=False, p=None)])

        self.path_identifier_list = []
        self.internal_node_index_list = []
        for leaf_index in tf.unstack(tf.constant([i for i in range(self.leaf_node_num_)])):
            for current_depth in tf.unstack(tf.constant([i for i in range(1, self.depth+1)])):
                path_identifier = tf.cast(tf.math.floormod(tf.math.floor(leaf_index/(tf.math.pow(2, (self.depth-current_depth)))), 2), tf.float32)
                internal_node_index =  tf.cast(tf.cast(tf.math.pow(2, (current_depth-1)), tf.float32) + tf.cast(tf.math.floor(leaf_index/(tf.math.pow(2, (self.depth-(current_depth-1))))), tf.float32) - 1.0, tf.int64)
                self.path_identifier_list.append(path_identifier)
                self.internal_node_index_list.append(internal_node_index)
        self.path_identifier_list = tf.reshape(tf.stack(self.path_identifier_list), (-1,self.depth))
        self.internal_node_index_list = tf.reshape(tf.cast(tf.stack(self.internal_node_index_list), tf.int64), (-1,self.depth))    

        leaf_classes_array_shape = [self.n_estimators,self.leaf_node_num_,] if self.number_of_classes == 2 or self.objective == 'regression' else [self.n_estimators, self.leaf_node_num_, self.number_of_classes]
        
        if self.estimator_leaf_weights:
            weight_shape = [self.n_estimators,self.leaf_node_num_]
        else:
            weight_shape = [1,self.n_estimators]

            
        self.estimator_weights = self.add_weight(name="estimator_weights", 
                                         initializer=tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + self.model_seed + 1}}),
                                         trainable=True,
                                         shape=weight_shape)
        self.split_values = self.add_weight(name="split_values",  
                                            initializer=tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + self.model_seed + 2}}),
                                            trainable=True,
                                            shape=[self.n_estimators,self.internal_node_num_, self.selected_variables])
        self.split_index_array = self.add_weight(name="split_index_array",
                                                 initializer=tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + self.model_seed + 3}}),
                                                 trainable=True,
                                                 shape=[self.n_estimators,self.internal_node_num_, self.selected_variables])
        self.leaf_classes_array = self.add_weight(name="leaf_classes_array",  
                                                 initializer=tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + self.model_seed + 4}}),
                                                 trainable=True,
                                                 shape=leaf_classes_array_shape)      


    def call(self, inputs, training):
        X = inputs


        if self.data_subset_fraction < 1.0 and training:
            X_estimator = tf.nn.embedding_lookup(tf.transpose(X), self.features_by_estimator)
            X_estimator = tf.transpose(tf.gather_nd(tf.transpose(X_estimator, [0,2,1]), tf.expand_dims(self.data_select,2), batch_dims=1), [1,0,2])
        else:
            X_estimator = tf.gather(X, self.features_by_estimator, axis=1)

        split_values_complete = self.split_values
        split_index_array_complete = self.split_index_array
        leaf_classes_array = self.leaf_classes_array    

        if self.normalize_index_array:
            mean = tf.stop_gradient(tf.reduce_mean(split_index_array_complete))
            std = tf.stop_gradient(tf.math.reduce_std(split_index_array_complete))
            split_index_array_complete = (split_index_array_complete - mean) / std        

        if self.split_index_activation == 'softmax':
            split_index_array_complete = tf.keras.activations.softmax(split_index_array_complete)
        elif self.split_index_activation == 'entmax':
            split_index_array_complete = entmax15(split_index_array_complete)
        elif self.split_index_activation == 'sparsemax':
            split_index_array_complete = tfa.activations.sparsemax(split_index_array_complete)                   

        
        adjust_constant = tf.stop_gradient(split_index_array_complete - tfa.seq2seq.hardmax(split_index_array_complete))
        split_index_array_complete = split_index_array_complete - adjust_constant        

        split_index_array_complete_selected = tf.gather(split_index_array_complete, self.internal_node_index_list, axis=1)
        split_values_complete_selected = tf.gather(split_values_complete, self.internal_node_index_list, axis=1)

        s1_sum = tf.einsum("ijkl,ijkl->ijk", split_values_complete_selected, split_index_array_complete_selected)
        s2_sum = tf.einsum("ijk,jlmk->ijlm", X_estimator, split_index_array_complete_selected)

        if self.split_decision_activation == 'softsign':
            node_result = (tf.nn.softsign(s1_sum-s2_sum) + 1) / 2
        elif self.split_decision_activation == 'entmoid':
            node_result = entmoid15_tf(s1_sum-s2_sum)               
        elif self.split_decision_activation == 'sigmoid':
            node_result = tf.sigmoid(s1_sum-s2_sum)               

        node_result_corrected = node_result - tf.stop_gradient(node_result - tf.round(node_result))

        p = tf.reduce_prod(((1-self.path_identifier_list)*node_result_corrected + self.path_identifier_list*(1-node_result_corrected)), axis=3)

        if self.estimator_leaf_weights:
            estimator_weights_leaf = tf.einsum("jk,ijk->ij", self.estimator_weights, p) 

            if self.weight_activation_st == 'softmax':
                estimator_weights_leaf_softmax = tf.keras.activations.softmax(estimator_weights_leaf)
            elif self.weight_activation_st == 'entmax':
                estimator_weights_leaf_softmax = entmax15(estimator_weights_leaf)
            elif self.weight_activation_st == 'sparsemax':
                estimator_weights_leaf_softmax = tfa.activations.sparsemax(estimator_weights_leaf)
            elif self.weight_activation_st == 'normalize':              
                estimator_weights_leaf = estimator_weights_leaf - tf.expand_dims(tf.reduce_min(estimator_weights_leaf, axis=-1), -1) + 1e-6          
                estimator_weights_leaf_softmax = estimator_weights_leaf / tf.expand_dims(tf.reduce_sum(estimator_weights_leaf, axis=-1), -1)
            else:
                estimator_weights_leaf_softmax = estimator_weights_leaf #, axis=0

            estimator_weights_leaf_softmax = self.apply_dropout_leaf(estimator_weights_leaf_softmax, training=training)        
        else:            
            if self.weight_activation_st == 'softmax':
                estimator_weights_softmax = tf.squeeze(tf.keras.activations.softmax(self.estimator_weights), axis=0) #, axis=0
            elif self.weight_activation_st == 'entmax':
                estimator_weights_softmax = tf.squeeze(entmax15(self.estimator_weights), axis=0) #, axis=0
            elif self.weight_activation_st == 'sparsemax':
                estimator_weights_softmax = tf.squeeze(tfa.activations.sparsemax(self.estimator_weights), axis=0) #, axis=0
            elif self.weight_activation_st == 'normalize':
                estimator_weights_softmax = tf.squeeze(self.estimator_weights) / tf.reduce_sum(self.estimator_weights)            
            else:
                estimator_weights_softmax = tf.squeeze(self.estimator_weights, axis=0) #, axis=0

            estimator_weights_softmax = self.apply_dropout(estimator_weights_softmax, training=training)        



        if self.data_subset_fraction < 1.0 and training:
            if self.objective == 'regression' or self.number_of_classes == 2:                
                if self.logit_weights:
                    if self.objective == 'regression':
                        layer_output = tf.einsum('jk,ijk->ij', leaf_classes_array, p)
                    else:
                        layer_output = tf.math.sigmoid(tf.einsum('jk,ijk->ij', leaf_classes_array, p))
                    if self.estimator_leaf_weights:
                        layer_output = tf.einsum('ij,ij->ij', estimator_weights_leaf_softmax, layer_output)
                    else:
                        layer_output = tf.einsum('j,ij->ij', estimator_weights_softmax, layer_output)
                else:
                    layer_output = tf.einsum('jk,ijk->ij', leaf_classes_array, p)
                    if self.estimator_leaf_weights:
                        layer_output = tf.einsum('ij,ij->ij', estimator_weights_leaf_softmax, layer_output)
                    else:
                        layer_output = tf.einsum('j,ij->ij', estimator_weights_softmax, layer_output) 
            else:
                layer_output = None                                

        else:                 
            if self.objective == 'regression' or self.number_of_classes == 2:                
                if self.logit_weights:
                    if self.estimator_leaf_weights:
                        layer_output = tf.math.sigmoid(tf.einsum('jk,ijk->ij', leaf_classes_array, p))
                        layer_output = tf.einsum('ij,ij->ij', estimator_weights_leaf_softmax, layer_output)
                    else:
                        layer_output = tf.math.sigmoid(tf.einsum('jk,ijk->ij', leaf_classes_array, p))
                        layer_output = tf.einsum('j,ij->ij', estimator_weights_softmax, layer_output)
                else:
                    if self.estimator_leaf_weights:
                        layer_output = tf.einsum('ij,ijk->ij', estimator_weights_leaf_softmax, leaf_classes_array * p)   
                    else:                        
                        layer_output = tf.einsum('j,ijk->ij', estimator_weights_softmax, leaf_classes_array * p)   
                        
            else:
                if self.logit_weights:
                    layer_output = tf.keras.activations.softmax(tf.einsum('jkl,ijk->ijl', leaf_classes_array, p))
                    if self.estimator_leaf_weights:
                        layer_output = tf.einsum('ij,ijl->ikl', estimator_weights_leaf_softmax, layer_output)
                    else:                         
                        layer_output = tf.einsum('j,ijl->ikl', estimator_weights_softmax, layer_output)
                else:
                    if self.estimator_leaf_weights:
                        layer_output = tf.einsum('ij,jkl,ijk->ikl', estimator_weights_leaf_softmax, leaf_classes_array, p)
                    else:                        
                        layer_output = tf.einsum('j,jkl,ijk->ikl', estimator_weights_softmax, leaf_classes_array, p)


        return layer_output

    def output_with_weights(self, inputs, training=False):
        X = inputs

        if self.data_subset_fraction < 1.0 and training:
            X_estimator = tf.nn.embedding_lookup(tf.transpose(X), self.features_by_estimator)
            X_estimator = tf.transpose(tf.gather_nd(tf.transpose(X_estimator, [0,2,1]), tf.expand_dims(self.data_select,2), batch_dims=1), [1,0,2])
        else:
            X_estimator = tf.gather(X, self.features_by_estimator, axis=1)

        split_values_complete = self.split_values
        split_index_array_complete = self.split_index_array
        leaf_classes_array = self.leaf_classes_array    

        if self.normalize_index_array:
            mean = tf.stop_gradient(tf.reduce_mean(split_index_array_complete))
            std = tf.stop_gradient(tf.math.reduce_std(split_index_array_complete))
            split_index_array_complete = (split_index_array_complete - mean) / std        

        if self.split_index_activation == 'softmax':
            split_index_array_complete = tf.keras.activations.softmax(split_index_array_complete)
        elif self.split_index_activation == 'entmax':
            split_index_array_complete = entmax15(split_index_array_complete)
        elif self.split_index_activation == 'sparsemax':
            split_index_array_complete = tfa.activations.sparsemax(split_index_array_complete)                    

        
        adjust_constant = tf.stop_gradient(split_index_array_complete - tfa.seq2seq.hardmax(split_index_array_complete))
        split_index_array_complete = split_index_array_complete - adjust_constant        

        split_index_array_complete_selected = tf.gather(split_index_array_complete, self.internal_node_index_list, axis=1)
        split_values_complete_selected = tf.gather(split_values_complete, self.internal_node_index_list, axis=1)

        s1_sum = tf.einsum("ijkl,ijkl->ijk", split_values_complete_selected, split_index_array_complete_selected)
        s2_sum = tf.einsum("ijk,jlmk->ijlm", X_estimator, split_index_array_complete_selected)

        if self.split_decision_activation == 'softsign':
            node_result = (tf.nn.softsign(s1_sum-s2_sum) + 1) / 2
        elif self.split_decision_activation == 'entmoid':
            node_result = entmoid15_tf(s1_sum-s2_sum)               
        elif self.split_decision_activation == 'sigmoid':
            node_result = tf.sigmoid(s1_sum-s2_sum)               

        node_result_corrected = node_result - tf.stop_gradient(node_result - tf.round(node_result))

        p = tf.reduce_prod(((1-self.path_identifier_list)*node_result_corrected + self.path_identifier_list*(1-node_result_corrected)), axis=3)

        if self.estimator_leaf_weights:
            estimator_weights_leaf = tf.einsum("jk,ijk->ij", self.estimator_weights, p) 

            if self.weight_activation_st == 'softmax':
                estimator_weights_leaf_softmax = tf.keras.activations.softmax(estimator_weights_leaf)
            elif self.weight_activation_st == 'entmax':
                estimator_weights_leaf_softmax = entmax15(estimator_weights_leaf)
            elif self.weight_activation_st == 'sparsemax':
                estimator_weights_leaf_softmax = tfa.activations.sparsemax(estimator_weights_leaf)
            elif self.weight_activation_st == 'normalize':              
                estimator_weights_leaf = estimator_weights_leaf - tf.expand_dims(tf.reduce_min(estimator_weights_leaf, axis=-1), -1) + 1e-6          
                estimator_weights_leaf_softmax = estimator_weights_leaf / tf.expand_dims(tf.reduce_sum(estimator_weights_leaf, axis=-1), -1)
            else:
                estimator_weights_leaf_softmax = estimator_weights_leaf #, axis=0

            estimator_weights_leaf_softmax = self.apply_dropout_leaf(estimator_weights_leaf_softmax, training=training)        

            weigths = estimator_weights_leaf_softmax
        else:            
            if self.weight_activation_st == 'softmax':
                estimator_weights_softmax = tf.squeeze(tf.keras.activations.softmax(self.estimator_weights), axis=0) #, axis=0
            elif self.weight_activation_st == 'entmax':
                estimator_weights_softmax = tf.squeeze(entmax15(self.estimator_weights), axis=0) #, axis=0
            elif self.weight_activation_st == 'sparsemax':
                estimator_weights_softmax = tf.squeeze(tfa.activations.sparsemax(self.estimator_weights), axis=0) #, axis=0
            elif self.weight_activation_st == 'normalize':
                estimator_weights_softmax = tf.squeeze(self.estimator_weights) / tf.reduce_sum(self.estimator_weights)            
            else:
                estimator_weights_softmax = tf.squeeze(self.estimator_weights, axis=0) #, axis=0

            estimator_weights_softmax = self.apply_dropout(estimator_weights_softmax, training=training)        

            weigths = estimator_weights_softmax

        if self.data_subset_fraction < 1.0 and training:
            if self.objective == 'regression' or self.number_of_classes == 2:                
                if self.logit_weights:
                    if self.objective == 'regression':
                        layer_output = tf.einsum('jk,ijk->ij', leaf_classes_array, p)
                    else:
                        layer_output = tf.math.sigmoid(tf.einsum('jk,ijk->ij', leaf_classes_array, p))
                    if self.estimator_leaf_weights:
                        layer_output = tf.einsum('ij,ij->ij', estimator_weights_leaf_softmax, layer_output)
                    else:
                        layer_output = tf.einsum('j,ij->ij', estimator_weights_softmax, layer_output)
                else:
                    layer_output = tf.einsum('jk,ijk->ij', leaf_classes_array, p)
                    if self.estimator_leaf_weights:
                        layer_output = tf.einsum('ij,ij->ij', estimator_weights_leaf_softmax, layer_output)
                    else:
                        layer_output = tf.einsum('j,ij->ij', estimator_weights_softmax, layer_output) 
            else:
                layer_output = None                                

        else:                 
            if self.objective == 'regression' or self.number_of_classes == 2:                
                if self.logit_weights:
                    if self.estimator_leaf_weights:
                        layer_output = tf.math.sigmoid(tf.einsum('jk,ijk->ij', leaf_classes_array, p))
                        layer_output = tf.einsum('ij,ij->ij', estimator_weights_leaf_softmax, layer_output)
                    else:
                        layer_output = tf.math.sigmoid(tf.einsum('jk,ijk->ij', leaf_classes_array, p))
                        layer_output = tf.einsum('j,ij->ij', estimator_weights_softmax, layer_output)
                else:
                    if self.estimator_leaf_weights:
                        layer_output = tf.einsum('ij,ijk->ij', estimator_weights_leaf_softmax, leaf_classes_array * p)   
                    else:                        
                        layer_output = tf.einsum('j,ijk->ij', estimator_weights_softmax, leaf_classes_array * p)   
                        
            else:
                if self.logit_weights:
                    layer_output = tf.keras.activations.softmax(tf.einsum('jkl,ijk->ijl', leaf_classes_array, p))
                    if self.estimator_leaf_weights:
                        layer_output = tf.einsum('ij,ijl->ikl', estimator_weights_leaf_softmax, layer_output)
                    else:                         
                        layer_output = tf.einsum('j,ijl->ikl', estimator_weights_softmax, layer_output)
                else:
                    if self.estimator_leaf_weights:
                        layer_output = tf.einsum('ij,jkl,ijk->ikl', estimator_weights_leaf_softmax, leaf_classes_array, p)
                    else:                        
                        layer_output = tf.einsum('j,jkl,ijk->ikl', estimator_weights_softmax, leaf_classes_array, p)


        return layer_output, weigths

    def apply_dropout(self,
                      index_array: tf.Tensor,
                      training: bool):

        if training and self.dropout > 0.0:
            
            index_array = tf.nn.dropout(index_array, rate=self.dropout)*(1-self.dropout)
            index_array = index_array/tf.reduce_sum(index_array)
        else:
            index_array = index_array
            
        return index_array    
    
    
    def apply_dropout_leaf(self,
                      index_array: tf.Tensor,
                      training: bool):

        if training and self.dropout > 0.0:           
            #index_array = tf.nn.dropout(index_array, rate=self.dropout, noise_shape=[tf.shape(index_array)[1]])*(1-self.dropout)
            index_array = tf.nn.dropout(index_array, rate=self.dropout)*(1-self.dropout)
            index_array = index_array/tf.expand_dims(tf.reduce_sum(index_array, axis=1), 1)
        else:
            index_array = index_array
            
        return index_array       
        
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)    
    

def make_batch(iterable, n=1, random_seed=42):
    tf.random.set_seed(random_seed)
    iterable = tf.random.shuffle(value=iterable, seed=random_seed)
    #rng = np.random.default_rng(seed=random_seed)
    #rng.shuffle(iterable)
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]   
        
def my_gather_nd(params, indices):
    idx_shape = tf.shape(indices)
    params_shape = tf.shape(params)
    idx_dims = idx_shape[-1]
    gather_shape = params_shape[idx_dims:]
    params_flat = tf.reshape(params, tf.concat([[-1], gather_shape], axis=0))
    axis_step = tf.cast(tf.math.cumprod(params_shape[:idx_dims], exclusive=True, reverse=True), tf.int64)
    indices_flat = tf.reduce_sum(indices * axis_step, axis=-1)
    result_flat = tf.gather(params_flat, indices_flat)
    return tf.reshape(result_flat, tf.concat([idx_shape[:-1], gather_shape], axis=0))
        
def make_batch_det(iterable, n=1):
    l = iterable.shape[0]
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]   

         
def entmax15(inputs, axis=-1):
    """
    Entmax 1.5 implementation, heavily inspired by
     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs
    """
    @tf.custom_gradient
    def _entmax_inner(inputs):
        with tf.name_scope('entmax'):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            inputs -= tf.reduce_max(inputs, axis, keepdims=True)  # subtract max for stability

            threshold, _ = entmax_threshold_and_support(inputs, axis)
            outputs_sqrt = tf.nn.relu(inputs - threshold)
            outputs = tf.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with tf.name_scope('entmax_grad'):
                d_inputs = d_outputs * outputs_sqrt
                q = tf.reduce_sum(d_inputs, axis=axis, keepdims=True) 
                q = q / tf.reduce_sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs
    
        return outputs, grad_fn
    
    return _entmax_inner(inputs)


@tf.custom_gradient
def sparse_entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param labels: reference answers vector int64[batch_size] \in [0, num_classes)
    :param logits: output matrix float32[batch_size, num_classes] (not actually logits :)
    :returns: elementwise loss, float32[batch_size]
    """
    assert logits.shape.ndims == 2 and labels.shape.ndims == 1
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - tf.one_hot(labels, depth=tf.shape(logits)[-1], axis=-1)
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)
    
    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


@tf.custom_gradient
def entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param logits: "logits" matrix float32[batch_size, num_classes]
    :param labels: reference answers indicators, float32[batch_size, num_classes]
    :returns: elementwise loss, float32[batch_size]
    
    WARNING: this function does not propagate gradients through :labels:
    This behavior is the same as like softmax_crossentropy_with_logits v1
    It may become an issue if you do something like co-distillation
    """
    assert labels.shape.ndims == logits.shape.ndims == 2
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - labels
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)

    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


def top_k_over_axis(inputs, k, axis=-1, **kwargs):
    """ performs tf.nn.top_k over any chosen axis """
    with tf.name_scope('top_k_along_axis'):
        if axis == -1:
            return tf.nn.top_k(inputs, k, **kwargs)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = tf.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = tf.nn.top_k(
            input_perm, k=k, **kwargs)

        input_sorted = tf.transpose(input_perm_sorted, inv_order)
        sort_indices = tf.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """ creates indices 0, ... , input[axis] unsqueezed to input dimensios """
    assert inputs.shape.ndims is not None
    rho = tf.cast(tf.range(1, tf.shape(inputs)[axis] + 1), dtype=inputs.dtype)
    view = [1] * inputs.shape.ndims
    view[axis] = -1
    return tf.reshape(rho, view)


def gather_over_axis(values, indices, gather_axis):
    """
    replicates the behavior of torch.gather for tf<=1.8;
    for newer versions use tf.gather with batch_dims
    :param values: tensor [d0, ..., dn]
    :param indices: int64 tensor of same shape as values except for gather_axis
    :param gather_axis: performs gather along this axis
    :returns: gathered values, same shape as values except for gather_axis
        If gather_axis == 2
        gathered_values[i, j, k, ...] = values[i, j, indices[i, j, k, ...], ...]
        see torch.gather for more detils
    """
    assert indices.shape.ndims is not None
    assert indices.shape.ndims == values.shape.ndims

    ndims = indices.shape.ndims
    gather_axis = gather_axis % ndims
    shape = tf.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = tf.range(tf.cast(shape[axis_i], dtype=indices.dtype), dtype=indices.dtype)
            index_i = tf.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = tf.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)

    return tf.gather_nd(values, tf.stack(selectors, axis=-1))


def entmax_threshold_and_support(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """

    with tf.name_scope('entmax_threshold_and_support'):
        num_outcomes = tf.shape(inputs)[axis]
        inputs_sorted, _ = top_k_over_axis(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_like(inputs, axis=axis)

        mean = tf.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = tf.cumsum(tf.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - tf.square(mean))) / rho

        delta_nz = tf.nn.relu(delta)
        tau = mean - tf.sqrt(delta_nz)

        support_size = tf.reduce_sum(tf.cast(tf.less_equal(tau, inputs_sorted), tf.int64), axis=axis, keepdims=True)

        tau_star = gather_over_axis(tau, support_size - 1, axis)
    return tau_star, support_size



class F1ScoreSparse(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average='macro', threshold=0.5, name='f1score_sparse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.metric = tfa.metrics.F1Score(average=average, num_classes=1 if num_classes==2 else num_classes , threshold=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert sparse labels to dense format
        if self.num_classes > 2:
            y_true_dense = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=self.num_classes)
        else:
            y_true_dense = y_true
        # Update precision and recall
        self.metric.update_state(y_true_dense, y_pred, sample_weight)

    def result(self):
        f1_score = self.metric.result()
        return f1_score

    def reset_states(self):
        self.metric.reset_states()
        
    def get_config(self):
        config = super(F1ScoreSparse, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'metric': self.metric,
        })
        
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)           



class R2ScoreTransform(tf.keras.metrics.Metric):
    def __init__(self, transformation_type=None, mean=0, std=1, name='r2score_transform', **kwargs):
        super().__init__(name=name, **kwargs)
        self.transformation_type = transformation_type
        self.mean = mean
        self.std = std
        self.metric = tfa.metrics.RSquare()

    def update_state(self, y_true, y_pred, sample_weight=None):

        if not tf.keras.backend.learning_phase():
            if self.transformation_type == 'mean':
                y_true = (y_true - self.mean) / self.std
            elif self.transformation_type == 'log':
                y_true = tf.math.log(y_true+1e-7)   
            
        # Update precision and recall
        self.metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        r2_score = self.metric.result()
        return r2_score

    def reset_states(self):
        self.metric.reset_states()        
        
        

def loss_function_regression(loss_name, mean, std, transformation_type='mean'): #mean, log, 
    loss_function = tf.keras.losses.get(loss_name)                                   
    def _loss_function_regression(y_true, y_pred):
        #if tf.keras.backend.learning_phase():
        if transformation_type == 'mean':
            y_true = (y_true - mean) / std
        elif transformation_type == 'log':
            y_true = tf.math.log(y_true+1e-7)

        loss = loss_function(y_true, y_pred)
        
        return loss
    return _loss_function_regression
    

def softsign(x):
    return x / (1 + tf.abs(x))

def softsignmax(x):
    softsign_values = softsign(x)
    return softsign_values / tf.expand_dims(tf.reduce_sum(softsign_values,axis=-1), -1)


def softsignmax_v2(x):
    # Compute the sum of absolute values
    S = tf.expand_dims(tf.reduce_sum(x,axis=-1), -1)#tf.reduce_sum(tf.abs(x))
    
    # Normalize the vector
    x_norm = tf.divide(x, S)
    
    # Apply softsign element-wise
    y = tf.divide(x_norm, (1 + tf.abs(x_norm)))
    
    return y

import tensorflow as tf

def _threshold_and_support(input, dim=-1):
    Xsrt = tf.sort(input, axis=dim, direction='DESCENDING')

    rho = tf.range(1, tf.shape(input)[dim] + 1, dtype=input.dtype)
    mean = tf.math.cumsum(Xsrt, axis=dim) / rho
    mean_sq = tf.math.cumsum(tf.square(Xsrt), axis=dim) / rho
    ss = rho * (mean_sq - tf.square(mean))
    delta = (1 - ss) / rho

    delta_nz = tf.maximum(delta, 0)
    tau = mean - tf.sqrt(delta_nz)

    support_size = tf.reduce_sum(tf.cast(tau <= Xsrt, tf.int32), axis=dim)
    tau_star = tf.gather(tau, support_size - 1, batch_dims=-1)
    return tau_star, support_size

def entmax15_tf(input, dim=-1):
    max_val = tf.reduce_max(input, axis=dim, keepdims=True)
    input = input - max_val
    input = input / 2

    tau_star, _ = _threshold_and_support(input, dim=dim)
    output = tf.pow(tf.clip_by_value(input - tau_star, 0, float('inf')), 2)
    return output

def entmoid15_tf(input):
    input_abs = tf.abs(input)
    tau = (input_abs + tf.sqrt(tf.nn.relu(8 - tf.square(input_abs)))) / 2
    mask = tau <= input
    tau = tf.where(mask, 2.0, tau)
    y_neg = 0.25 * tf.square(tf.nn.relu(tau - input_abs))
    return tf.where(input >= 0, 1 - y_neg, y_neg)
