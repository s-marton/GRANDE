from models.basemodel import BaseModel
from utils.io_utils import get_output_path

import numpy as np
import os

import tensorflow as tf
from sklearn.metrics import log_loss

from models.dnf_lib.DNFNet.ModelHandler import ModelHandler, EarlyStopping, ReduceLRonPlateau
from models.dnf_lib.config import get_config
from models.dnf_lib.Utils.NumpyGenerator import NumpyGenerator
from models.dnf_lib.Utils.experiment_utils import create_model, create_experiment_directory


'''
    Net-DNF: Effective Deep Modeling of Tabular Data  (https://openreview.net/forum?id=73WTGs96kho)
    
    Code adapted from: https://github.com/amramabutbul/DisjunctiveNormalFormNet
'''


class GDF(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.config, self.score_config = get_config(args.dataset, 'GDF')

        self.config.update({
            'input_dim': args.num_features,
            'output_dim': args.num_classes,
            'translate_label_to_one_hot': True if args.objective == "classification" else False,
            'epochs': args.epochs,
            'early_stopping_patience': args.early_stopping_rounds,
            'batch_size': args.batch_size,
            'GPU': str(args.gpu_ids),
            **self.params
        })

        self.score_config.update({
            'score_metric': log_loss,
            'score_increases': False,
        })

        print(self.config)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['GPU']
        tf.reset_default_graph()
        tf.random.set_random_seed(seed=self.config['random_seed'])
        np.random.seed(seed=self.config['random_seed'])
        self.score_metric = self.score_config['score_metric']

        self.experiment_dir, self.weights_dir, self.logs_dir = create_experiment_directory(self.config,
                                                                                           return_sub_dirs=True)
        
        self.model = create_model(self.config, models_module_name=self.config['models_module_name'])

        self.model_handler = None

    def fit(self, X, y, X_val=None, y_val=None):


        return loss, val_loss

    def predict_proba(self, X):

        self.prediction_probabilities = np.concatenate(y_pred_sorted, axis=0).reshape(-1, self.args.num_classes)

        if self.args.objective == "binary":
            self.prediction_probabilities = np.concatenate((1 - self.prediction_probabilities,
                                                            self.prediction_probabilities), 1)

        return self.prediction_probabilities

    def save_model(self, filename_extension="", directory="models"):
        pass

    def get_model_size(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print(total_parameters)

        return total_parameters


    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
        }
        return params
