import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa

import sklearn
from copy import deepcopy
import category_encoders as ce
import pandas as pd
import math
from focal_loss import SparseCategoricalFocalLoss

import pickle
import zipfile
import os
     
class GRANDE(tf.keras.Model):
    def __init__(self, 
                 params, 
                 args):
        
        params.update(args)
        self.config = None

        super(GRANDE, self).__init__()      
        self.set_params(**params)

        self.is_fitted = False

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        tf.keras.utils.set_random_seed(self.random_seed)

    def call(self, inputs, training):
        # einsum syntax:         
        #       - b is the batch size
        #       - e is the number of estimators
        #       - l the number of leaf nodes  (i.e. the number of paths)
        #       - i is the number of internal nodes
        #       - d is the depth (i.e. the length of each path)
        #       - n is the number of variables (one value is stored for each variable)        

        
        # Adjust data: For each estimator select a subset of the features (output shape: (b, e, n))
        if self.data_subset_fraction < 1.0 and training:
            #select subset of samples for each estimator during training if hyperparameter set accordingly
            X_estimator = tf.nn.embedding_lookup(tf.transpose(inputs), self.features_by_estimator)
            X_estimator = tf.transpose(tf.gather_nd(tf.transpose(X_estimator, [0,2,1]), tf.expand_dims(self.data_select,2), batch_dims=1), [1,0,2])
        else:
            #use complete data
            X_estimator = tf.gather(inputs, self.features_by_estimator, axis=1)

        #entmax transformaton
        split_index_array = entmax15TF(self.split_index_array)
  
        #use ST-Operator to get one-hot encoded vector for feature index
        split_index_array = split_index_array - tf.stop_gradient(split_index_array - tf.one_hot(tf.argmax(split_index_array, axis=-1), depth=split_index_array.shape[-1]))

        # as split_index_array_selected is one-hot-encoded, taking the sum over the last axis after multiplication results in selecting the desired value at the index
        s1_sum = tf.einsum("ein,ein->ei", self.split_values, split_index_array)
        s2_sum = tf.einsum("ben,ein->bei", X_estimator, split_index_array)

        # calculate the split (output shape: (b, e, i))
        node_result = (tf.nn.softsign(s1_sum-s2_sum) + 1) / 2
        
        #use round operation with ST operator to get hard decision for each node
        node_result_corrected = node_result - tf.stop_gradient(node_result - tf.round(node_result))

        #generate tensor for further calculation: 
        # - internal_node_index_list holds the indices for the internal nodes traversed for each path (there are l paths) in the tree
        # - for each estimator and for each path in each estimator, the tensors hold the information for all internal nodex traversed
        # - the resulting shape of the tensors is (b, e, l, d):        
        node_result_extended = tf.gather(node_result_corrected, self.internal_node_index_list, axis=2)

            
        #reduce the path via multiplication to get result for each path (in each estimator) based on the results of the corresponding internal nodes (output shape: (b, e, l))
        p = tf.reduce_prod(((1-self.path_identifier_list)*node_result_extended + self.path_identifier_list*(1-node_result_extended)), axis=3)

        #calculate instance-wise leaf weights for each estimator by selecting the weight of the selected path for each estimator
        estimator_weights_leaf = tf.einsum("el,bel->be", self.estimator_weights, p) 

        #use softmax over weights for each instance
        estimator_weights_leaf_softmax = tf.keras.activations.softmax(estimator_weights_leaf)

        #optional dropout (deactivating random estimators)
        estimator_weights_leaf_softmax = self.apply_dropout_leaf(estimator_weights_leaf_softmax, training=training)        

        #get raw prediction for each estimator
        #optionally transform to probability distribution before weighting
        if self.objective == 'regression':
            layer_output = tf.einsum('el,bel->be', self.leaf_classes_array, p)                
            layer_output = tf.einsum('be,be->be', estimator_weights_leaf_softmax, layer_output)
        elif self.objective == 'binary':
            if self.from_logits:
                layer_output = tf.einsum('el,bel->be', self.leaf_classes_array, p)               
            else:
                layer_output = tf.math.sigmoid(tf.einsum('el,bel->be', self.leaf_classes_array, p))
            layer_output = tf.einsum('be,be->be', estimator_weights_leaf_softmax, layer_output)             
        elif self.objective == 'classification':
            if self.from_logits:
                layer_output = tf.einsum('elc,bel->bec', self.leaf_classes_array, p)
                layer_output = tf.einsum('be,bec->bec', estimator_weights_leaf_softmax, layer_output)
            else:
                layer_output = tf.keras.activations.softmax(tf.einsum('elc,bel->bec', self.leaf_classes_array, p))
                layer_output = tf.einsum('be,bec->bec', estimator_weights_leaf_softmax, layer_output)

        if self.data_subset_fraction < 1.0 and training:
            result = tf.scatter_nd(indices=tf.expand_dims(self.data_select, 2), updates=tf.transpose(layer_output), shape=[tf.shape(inputs)[0]])
            result = (result / self.counts) * self.n_estimators   
        else:
            if self.objective == 'regression' or self.objective == 'binary':                                   
                result = tf.einsum('be->b', layer_output)
            else:                    
                result = tf.einsum('bec->bc', layer_output)

        if self.objective == 'regression' or self.objective == 'binary':   
            result = tf.expand_dims(result, 1)

        return result

    def apply_preprocessing(self, X):
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if len(self.num_columns) > 0:
            X[self.num_columns] = X[self.num_columns].fillna(self.mean_train_num)
        if len(self.cat_columns) > 0:
            X[self.cat_columns] = X[self.cat_columns].fillna(self.mode_train_cat)        
        
        X = self.encoder_ordinal.transform(X)
        X = self.encoder_loo.transform(X)
        X = self.encoder_ohe.transform(X)

        X = self.normalizer.transform(X.values.astype(np.float64))

        return X
        
    def perform_preprocessing(self, 
                        X_train, 
                        y_train, 
                        X_val,
                        y_val):
   
        if isinstance(y_train, pd.Series):
            try:
                y_train = y_train.values.codes.astype(np.float64)
            except:
                pass
        if isinstance(y_val, pd.Series):
            try:
                y_val = y_val.values.codes.astype(np.float64)
            except:
                pass

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val)

        self.mean = np.mean(y_train)
        self.std = np.std(y_train)
                             
        binary_indices = []
        low_cardinality_indices = []
        high_cardinality_indices = []
        num_columns = []
        for column_index, column in enumerate(X_train.columns):
            if column_index in self.cat_idx:
                if len(X_train.iloc[:,column_index].unique()) <= 2:
                    binary_indices.append(column)
                elif len(X_train.iloc[:,column_index].unique()) < 5:
                    low_cardinality_indices.append(column)
                else:
                    high_cardinality_indices.append(column)
            else:
                num_columns.append(column)     

        cat_columns = [col for col in X_train.columns if col not in num_columns]

        if len(num_columns) > 0:
            self.mean_train_num = X_train[num_columns].mean(axis=0).iloc[0]
            X_train[num_columns] = X_train[num_columns].fillna(self.mean_train_num)
            X_val[num_columns] = X_val[num_columns].fillna(self.mean_train_num)
        if len(cat_columns) > 0:
            self.mode_train_cat = X_train[cat_columns].mode(axis=0).iloc[0]
            X_train[cat_columns] = X_train[cat_columns].fillna(self.mode_train_cat)
            X_val[cat_columns] = X_val[cat_columns].fillna(self.mode_train_cat)

        self.cat_columns = cat_columns
        self.num_columns = num_columns
        
        self.encoder_ordinal = ce.OrdinalEncoder(cols=binary_indices)
        self.encoder_ordinal.fit(X_train)
        X_train = self.encoder_ordinal.transform(X_train)
        X_val = self.encoder_ordinal.transform(X_val)     
        
        self.encoder_loo = ce.LeaveOneOutEncoder(cols=high_cardinality_indices)
        if self.objective == 'regression':
            self.encoder_loo.fit(X_train, (y_train-self.mean)/self.std)
        else:
            self.encoder_loo.fit(X_train, y_train)                                 
        X_train = self.encoder_loo.transform(X_train)
        X_val = self.encoder_loo.transform(X_val)
        
        self.encoder_ohe = ce.OneHotEncoder(cols=low_cardinality_indices)
        self.encoder_ohe.fit(X_train)
        X_train = self.encoder_ohe.transform(X_train)
        X_val = self.encoder_ohe.transform(X_val)

        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        
        quantile_noise = 1e-4
        quantile_train = np.copy(X_train.values).astype(np.float64)
        np.random.seed(42)
        stds = np.std(quantile_train, axis=0, keepdims=True)
        noise_std = quantile_noise / np.maximum(stds, quantile_noise)
        quantile_train += noise_std * np.random.randn(*quantile_train.shape)    

        quantile_train = pd.DataFrame(quantile_train, columns=X_train.columns, index=X_train.index)

        self.normalizer = sklearn.preprocessing.QuantileTransformer(
                                                                    n_quantiles=min(quantile_train.shape[0], 1000),
                                                                    output_distribution='normal',
                                                                    )

        self.normalizer.fit(quantile_train.values.astype(np.float64))
        X_train = self.normalizer.transform(X_train.values.astype(np.float64))
        X_val = self.normalizer.transform(X_val.values.astype(np.float64))
    
        return X_train, y_train, X_val, y_val
        
    def convert_to_numpy(self,data):
        """
        Converts input data (Pandas DataFrame, TensorFlow tensor, PyTorch tensor, list, or similar iterable) to a NumPy array.
    
        Args:
        data: Input data to be converted. Can be a Pandas DataFrame, TensorFlow tensor, list, or similar iterable.
    
        Returns:
        numpy_array: A NumPy array representation of the input data.
        """
        # Check if the data is a Pandas DataFrame
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        
        # Check if the data is a TensorFlow tensor
        elif isinstance(data, tf.Tensor):
            return data.numpy()
        
        # Check if the data is a list or similar iterable (not including strings)
        elif isinstance(data, (list, tuple, np.ndarray)):
            return np.array(data)
        
        else:
            raise TypeError("The input data type is not supported for conversion.")
    
    def fit(self, 
            X_train, 
            y_train, 
            X_val=None,
            y_val=None,
            **kwargs):

        if self.preprocess_data:
            X_train, y_train, X_val, y_val = self.perform_preprocessing(X_train, y_train, X_val, y_val)
        else:
            X_train = self.convert_to_numpy(X_train)
            y_train = self.convert_to_numpy(y_train)
            X_val = self.convert_to_numpy(X_val)
            y_val = self.convert_to_numpy(y_val)

        jit_compile = True #X_train.shape[0] < 10_000
        
        self.number_of_variables = X_train.shape[1]
        if self.use_class_weights:
            if self.objective == 'classification' or self.objective == 'binary':
                self.number_of_classes = len(np.unique(y_train))
                self.class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
    
                self.class_weight_dict = {}
                for i in range(self.number_of_classes):
                    self.class_weight_dict[i] = self.class_weights[i]
    
            else:
                self.number_of_classes = 1
                self.class_weights = np.ones_like(np.unique(y_train))
                self.class_weight_dict = None
        else:   
            if self.objective == 'classification' or self.objective == 'binary':
                self.number_of_classes = len(np.unique(y_train))
            else:
                self.number_of_classes = 1
            self.class_weights = np.ones_like(np.unique(y_train))
            self.class_weight_dict = None        
        
        self.build_model()

        self.compile(loss=self.loss_name, metrics=[], jit_compile=jit_compile, mean=self.mean, std=self.std, class_weight=self.class_weights)

        
        train_data = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(tf.convert_to_tensor(X_train), tf.float32),
                                                         tf.dtypes.cast(tf.convert_to_tensor(y_train), tf.float32)))

        if self.data_subset_fraction < 1.0:
            train_data = (train_data
                    #.shuffle(32_768)
                    .cache()
                    .batch(batch_size=self.batch_size, drop_remainder=True) 
                    .prefetch(tf.data.AUTOTUNE)      
                         )    
        else:
            train_data = (train_data
                    .shuffle(32_768)
                    .cache()
                    .batch(batch_size=self.batch_size, drop_remainder=False) 
                    .prefetch(tf.data.AUTOTUNE)      
                        )

        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_data = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(tf.convert_to_tensor(validation_data[0]), tf.float32), 
                                                             tf.dtypes.cast(tf.convert_to_tensor(validation_data[1]), tf.float32)))


            validation_data = (validation_data
                    .cache()
                    .batch(batch_size=self.batch_size, drop_remainder=False) 
                    .prefetch(tf.data.AUTOTUNE)      
                         )   

            monitor = 'val_loss'    
        else:
            monitor = 'loss'


        if 'callbacks' not in kwargs.keys():
            callbacks = []

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                          patience=self.early_stopping_epochs, 
                                                          min_delta=1e-3,
                                                          restore_best_weights=True)
        callbacks.append(early_stopping)

        if 'reduce_lr' in kwargs.keys() and kwargs['reduce_lr']:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=self.early_stopping_epochs//3)
            callbacks.append(reduce_lr)

        super(GRANDE, self).fit(train_data,
                                validation_data = validation_data,
                                epochs = self.epochs,
                                callbacks = callbacks,
                                class_weight = self.class_weight_dict,
                                verbose=self.verbose,
                                **kwargs)
          
    def build_model(self):

        tf.keras.utils.set_random_seed(self.random_seed)
        
        if self.selected_variables > 1:
            self.selected_variables = min(self.selected_variables, self.number_of_variables)
        else:
            self.selected_variables = int(self.number_of_variables * self.selected_variables)
            self.selected_variables = min(self.selected_variables, 50)
            self.selected_variables = max(self.selected_variables, 10)
            self.selected_variables = min(self.selected_variables, self.number_of_variables)  
        if self.objective != 'binary':
            self.data_subset_fraction = 1.0      
        if self.data_subset_fraction < 1.0:
            self.subset_size = tf.cast(self.batch_size * self.data_subset_fraction, tf.int32)
            if self.bootstrap:
                self.data_select = tf.random.uniform(shape=(self.n_estimators, self.subset_size), minval=0, maxval=self.batch_size, dtype=tf.int32)
            else:
                    
                indices = [np.random.choice(self.batch_size, size=self.subset_size, replace=False) for _ in range(self.n_estimators)]
                self.data_select = tf.stack(indices)

            items, self.counts = np.unique(self.data_select, return_counts=True)
            self.counts = tf.constant(self.counts, dtype=tf.float32)        
        
        self.features_by_estimator = tf.stack([np.random.choice(self.number_of_variables, size=(self.selected_variables), replace=False, p=None) for _ in range(self.n_estimators)])

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

        leaf_classes_array_shape = [self.n_estimators,self.leaf_node_num_,] if self.objective == 'binary' or self.objective == 'regression' else [self.n_estimators, self.leaf_node_num_, self.number_of_classes]
        
        weight_shape = [self.n_estimators,self.leaf_node_num_]



        self.estimator_weights = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + 1}})(shape=weight_shape),
                                             
                                             trainable=True,
                                             name="estimator_weights"
                                            )
        self.split_values = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + 2}})(shape=[self.n_estimators,self.internal_node_num_, self.selected_variables]), 
                                             trainable=True,
                                             name="split_values"
                                       )
        
        self.split_index_array = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + 3}})(shape=[self.n_estimators,self.internal_node_num_, self.selected_variables]), 
                                             trainable=True,
                                             name="split_index_array"
                                            )
        
        self.leaf_classes_array = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + 4}})(shape=leaf_classes_array_shape), 
                                             trainable=True,
                                             name="leaf_classes_array"
                                             )
        

    def compile(self, 
        loss, 
        metrics, 
        jit_compile,
        **kwargs):

        if self.objective == 'classification':

            if loss == 'crossentropy':
                if not self.focal_loss:
                    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=self.from_logits) #tf.keras.losses.get('categorical_crossentropy')
                else:
                    loss_function = SparseCategoricalFocalLoss(gamma=2, class_weight=self.class_weights, from_logits=from_logits)
            else:
                loss_function = tf.keras.losses.get(loss)  
                try:
                     loss_function.from_logits = self.from_logits
                except:
                    pass        
        elif self.objective == 'binary':
            if loss == 'crossentropy':
                if not self.focal_loss:
                    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=self.from_logits) #tf.keras.losses.get('binary_crossentropy')
                else:
                    loss_function = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=2, from_logits=self.from_logits)
            else:
                loss_function = tf.keras.losses.get(loss)  
                try:
                     loss_function.from_logits = self.from_logits
                except:
                    pass   

        elif self.objective == 'regression':
            loss_function = loss_function_regression(loss_name=loss, mean=kwargs['mean'], std=kwargs['std'])

        loss_function = loss_function_weighting(loss_function, temp=self.temperature)
        self.weights_optimizer = get_optimizer_by_name(optimizer_name=self.optimizer_name, learning_rate=self.learning_rate_weights, warmup_steps=0, cosine_decay_steps=self.cosine_decay_steps)
        self.index_optimizer = get_optimizer_by_name(optimizer_name=self.optimizer_name, learning_rate=self.learning_rate_index, warmup_steps=0, cosine_decay_steps=self.cosine_decay_steps)
        self.values_optimizer = get_optimizer_by_name(optimizer_name=self.optimizer_name, learning_rate=self.learning_rate_values, warmup_steps=0, cosine_decay_steps=self.cosine_decay_steps)
        self.leaf_optimizer = get_optimizer_by_name(optimizer_name=self.optimizer_name, learning_rate=self.learning_rate_leaf, warmup_steps=0, cosine_decay_steps=self.cosine_decay_steps)

        super(GRANDE, self).compile(loss=loss_function, metrics=metrics, jit_compile=jit_compile)
        
    def train_step(self, data):
    
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        if not self.built:
            _ = self(x, training=True) 

        with tf.GradientTape() as weights_tape:
            with tf.GradientTape() as index_tape:
                with tf.GradientTape() as values_tape:
                    with tf.GradientTape() as leaf_tape:                        
                        weights_tape.watch(self.estimator_weights)          
                        index_tape.watch(self.split_index_array)
                        values_tape.watch(self.split_values)
                        leaf_tape.watch(self.leaf_classes_array)
                        
                        y_pred = self(x, training=True)
                        loss = self.compute_loss(x=None, y=y, y_pred=y_pred, sample_weight=sample_weight)

        weights_gradients = weights_tape.gradient(loss, [self.estimator_weights])
        self.weights_optimizer.apply_gradients(zip(weights_gradients, [self.estimator_weights]))       

        index_gradients = index_tape.gradient(loss, [self.split_index_array])
        self.index_optimizer.apply_gradients(zip(index_gradients, [self.split_index_array]))
         
        values_gradients = values_tape.gradient(loss, [self.split_values])
        self.values_optimizer.apply_gradients(zip(values_gradients, [self.split_values]))
        
        leaf_gradients = leaf_tape.gradient(loss, [self.leaf_classes_array])  
        self.leaf_optimizer.apply_gradients(zip(leaf_gradients, [self.leaf_classes_array]))        


        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)        

        return {m.name: m.result() for m in self.metrics}
 

    def predict(self, X):

        if self.preprocess_data:
            X = self.apply_preprocessing(X)
        else:
            X = self.convert_to_numpy(X)
        
        preds = super(GRANDE, self).predict(X, self.batch_size, verbose=0)
        preds = tf.convert_to_tensor(preds)
        if self.objective == 'regression':
            preds = preds * self.std + self.mean  
        else:
            if self.from_logits:
                if self.objective == 'binary':
                    preds = tf.math.sigmoid(preds)
                elif self.objective == 'classification':
                    preds = tf.keras.activations.softmax(preds) 

        if self.objective == 'binary':
            preds = tf.stack([1-tf.squeeze(preds), tf.squeeze(preds)], axis=-1)      

        return preds.numpy()

    def set_params(self, **kwargs): 
                
        if self.config is None:
            self.config = self.default_parameters()

        self.config.update(kwargs)

        if self.config['n_estimators'] == 1:
            self.config['selected_variables'] = 1.0
            self.config['data_subset_fraction'] = 1.0
            self.config['bootstrap'] = False
            self.config['dropout'] = 0.0

        if 'loss' not in self.config.keys():
            if self.config['objective'] == 'classification' or self.config['objective'] == 'binary':
                self.config['loss'] = 'crossentropy'
                self.config['focal_loss'] = False
            elif self.config['objective'] == 'regression':
                self.config['loss'] = 'mse'
                self.config['focal_loss'] = False

        self.config['optimizer_name'] = self.config.pop('optimizer')
        self.config['loss_name'] = self.config.pop('loss')

        for arg_key, arg_value in self.config.items():
            setattr(self, arg_key, arg_value)     
        
        tf.keras.utils.set_random_seed(self.random_seed)   
                 
                
    def get_params(self):
        return self.config    

    def define_trial_parameters(self, trial, args):
        params = {
            'depth': trial.suggest_int("depth", 3, 7),
            'n_estimators': trial.suggest_int("n_estimators", 512, 2048),

            'learning_rate_weights':  trial.suggest_float("learning_rate_weights", 0.0001, 0.25),
            'learning_rate_index': trial.suggest_float("learning_rate_index", 0.0001, 0.25),
            'learning_rate_values': trial.suggest_float("learning_rate_values", 0.0001, 0.25),
            'learning_rate_leaf': trial.suggest_float("learning_rate_leaf", 0.0001, 0.25),

            'cosine_decay_steps': trial.suggest_categorical("cosine_decay_steps", [0, 100, 1000]),
            
            'dropout': trial.suggest_categorical("dropout", [0, 0.25, 0.5]),

            'selected_variables': trial.suggest_categorical("selected_variables", [1.0, 0.75, 0.5]),
            'data_subset_fraction': trial.suggest_categorical("data_subset_fraction", [1.0, 0.8]),
        }

        try:
            if args['objective'] != 'regression':
                params['focal_loss'] = trial.suggest_categorical("focal_loss", [True, False])
                params['temperature'] = trial.suggest_categorical("temperature", [0, 0.25])
        except:
            if self.objective  != 'regression':
                params['focal_loss'] = trial.suggest_categorical("focal_loss", [True, False])
                params['temperature'] = trial.suggest_categorical("temperature", [0, 0.25])
        return params

    def get_random_parameters(self, seed):
        rs = np.random.RandomState(seed)
        params = {
            'depth': rs.randint(3, 7),
            'n_estimators': rs.randint(512, 2048),

            'learning_rate_weights':  rs.uniform(0.0001, 0.25),
            'learning_rate_index': rs.uniform(0.0001, 0.25),
            'learning_rate_values': rs.uniform(0.0001, 0.25),
            'learning_rate_leaf': rs.uniform(0.0001, 0.25),

            'cosine_decay_steps': rs.choice([0, 100, 1000], p=[0.5, 0.25, 0.25]),
            'dropout': rs.choice([0, 0.25]),

            'selected_variables': rs.choice([1.0, 0.75, 0.5]),
            'data_subset_fraction': rs.choice([1.0, 0.8]),

        }

        if self.objective != 'regression':
            params['focal_loss'] = rs.choice([True, False])
            params['temperature'] = rs.choice([1, 1/3, 1/5, 1/7, 1/9, 0], p=[0.1, 0.1, 0.1, 0.1, 0.1,0.5])

        return params

    def default_parameters(self):
        params = {
            'depth': 5,
            'n_estimators': 2048,

            'learning_rate_weights': 0.005,
            'learning_rate_index': 0.01,
            'learning_rate_values': 0.01,
            'learning_rate_leaf': 0.01,

            'optimizer': 'adam',
            'cosine_decay_steps': 0,
            'temperature': 0.0,

            'initializer': 'RandomNormal',

            'loss': 'crossentropy',
            'focal_loss': False,

            'from_logits': True,
            'use_class_weights': True,
            'preprocess_data': True,

            'dropout': 0.0,

            'selected_variables': 0.8,
            'data_subset_fraction': 1.0,
            'bootstrap': False,
        }        

        return params

    def save_model(self, save_path='model_gande'):
        config = {'params': {
                    'depth': self.depth,
                    'n_estimators': self.n_estimators,
    
                    'std': self.std,
                    'mean': self.mean,
                    'mode_train_cat': self.mode_train_cat,
                    'mean_train_num': self.mean_train_num,

                    'encoder_ordinal': self.encoder_ordinal,
                    'encoder_loo': self.encoder_loo,
                    'encoder_ohe': self.encoder_ohe,
                    'normalizer': self.normalizer,

                    'number_of_classes': self.number_of_classes,
                    'number_of_variables': self.number_of_variables,
            
                    'selected_variables': self.selected_variables,
                    'data_subset_fraction': self.data_subset_fraction,
                 },
                 'args': {
                    'objective': self.objective,
                    
                    'random_seed': self.random_seed,
                    'verbose': self.verbose,
                     
                 }
                 }
         
        temp_dir = 'temp_model'
        os.makedirs(temp_dir, exist_ok=True)
        
        np.savez(os.path.join(temp_dir, 'variables.npz'), 
                 estimator_weights=self.estimator_weights.numpy(), 
                 split_values=self.split_values.numpy(), 
                 split_index_array=self.split_index_array.numpy(), 
                 leaf_classes_array=self.leaf_classes_array.numpy())
        
        config_path = os.path.join(temp_dir, 'config.pkl')
        with open(config_path, 'wb') as config_file:
            pickle.dump(config, config_file)
        
        with zipfile.ZipFile(save_path, 'w') as model_zip:
            for dirname, _, filenames in os.walk(temp_dir):
                for filename in filenames:
                    file_path = os.path.join(dirname, filename)
                    model_zip.write(file_path, arcname=os.path.relpath(file_path, temp_dir))
        
        for dirname, _, filenames in os.walk(temp_dir):
            for filename in filenames:
                os.remove(os.path.join(dirname, filename))
        os.rmdir(temp_dir)
        
        print(f"Model and config saved to {save_path}")


    def load_model(load_path='model_gande'):
        temp_dir = 'temp_model'
        with zipfile.ZipFile(load_path, 'r') as model_zip:
            model_zip.extractall(temp_dir)

        config_path = os.path.join(temp_dir, 'config.pkl')
        with open(config_path, 'rb') as config_file:
            config = pickle.load(config_file)


        model = GRANDE(params=config['params'], args=config['args'])

        model.build_model()
        model.predict(np.random.uniform(0,1, (1,config['params']['number_of_variables'])))
        
        variables = np.load(os.path.join(temp_dir, 'variables.npz'))
        
        model.estimator_weights.assign(variables['estimator_weights'])
        model.split_values.assign(variables['split_values'])
        model.split_index_array.assign(variables['split_index_array'])
        model.leaf_classes_array.assign(variables['leaf_classes_array'])
        
        for dirname, _, filenames in os.walk(temp_dir):
            for filename in filenames:
                os.remove(os.path.join(dirname, filename))
        os.rmdir(temp_dir)
        
        print("Model and config loaded successfully.")
        
        return model

    def apply_dropout_leaf(self,
                      index_array: tf.Tensor,
                      training: bool):

        if training and self.dropout > 0.0:           
            index_array = tf.nn.dropout(index_array, rate=self.dropout)*(1-self.dropout)
            index_array = index_array/tf.expand_dims(tf.reduce_sum(index_array, axis=1), 1)
        else:
            index_array = index_array
            
        return index_array       
    
         
def entmax15TF(inputs, axis=-1):

    # Implementation taken from: https://github.com/deep-spin/entmax/tree/master/entmax 

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

            threshold, _ = entmax_threshold_and_supportTF(inputs, axis)
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


def top_k_over_axisTF(inputs, k, axis=-1, **kwargs):
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


def _make_ix_likeTF(inputs, axis=-1):
    """ creates indices 0, ... , input[axis] unsqueezed to input dimensios """
    assert inputs.shape.ndims is not None
    rho = tf.cast(tf.range(1, tf.shape(inputs)[axis] + 1), dtype=inputs.dtype)
    view = [1] * inputs.shape.ndims
    view[axis] = -1
    return tf.reshape(rho, view)


def gather_over_axisTF(values, indices, gather_axis):
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


def entmax_threshold_and_supportTF(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """

    with tf.name_scope('entmax_threshold_and_supportTF'):
        num_outcomes = tf.shape(inputs)[axis]
        inputs_sorted, _ = top_k_over_axisTF(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_likeTF(inputs, axis=axis)

        mean = tf.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = tf.cumsum(tf.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - tf.square(mean))) / rho

        delta_nz = tf.nn.relu(delta)
        tau = mean - tf.sqrt(delta_nz)

        support_size = tf.reduce_sum(tf.cast(tf.less_equal(tau, inputs_sorted), tf.int64), axis=axis, keepdims=True)

        tau_star = gather_over_axisTF(tau, support_size - 1, axis)
    return tau_star, support_size   
        

def loss_function_weighting(loss_function, temp=0.25): 

    # Implementation of "Stochastic Re-weighted Gradient Descent via Distributionally Robust Optimization" from https://arxiv.org/abs/2306.09222

    loss_function.reduction = tf.keras.losses.Reduction.NONE
    def _loss_function_weighting(y_true, y_pred):
        loss = loss_function(y_true, y_pred)

        if temp > 0:
            clamped_loss = tf.clip_by_value(loss, clip_value_min=float('-inf'), clip_value_max=temp)

            out = loss * tf.stop_gradient(tf.exp(clamped_loss / (temp + 1)))
        else:
            out = loss
        
        return tf.reduce_mean(out)
    return _loss_function_weighting


def loss_function_regression(loss_name, mean, std): #mean, log, 
    loss_function = tf.keras.losses.get(loss_name)                                   
    def _loss_function_regression(y_true, y_pred):
        #if tf.keras.backend.learning_phase():
        y_true = (y_true - mean) / std

        loss = loss_function(y_true, y_pred)
        
        return loss
    return _loss_function_regression

def _threshold_and_supportTF(input, dim=-1):
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

def get_optimizer_by_name(optimizer_name, learning_rate, warmup_steps, cosine_decay_steps):

    
    if cosine_decay_steps > 0:
        learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
                                                                            initial_learning_rate=learning_rate,
                                                                            first_decay_steps=cosine_decay_steps,
                                                                            #first_decay_steps=steps_per_epoch,
                                                                        )

    if optimizer_name== 'SWA' or optimizer_name== 'EMA':
        #optimizer = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), average_period=5)
        frequency = 10
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             use_ema = True,
                                             ema_momentum = 1/frequency,
                                             ema_overwrite_frequency = 1
                                            )
    else:
        optimizer = tf.keras.optimizers.get(optimizer_name)
        optimizer.learning_rate = learning_rate
                
    return optimizer


