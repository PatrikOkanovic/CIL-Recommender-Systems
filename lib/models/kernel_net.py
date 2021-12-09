'''
Adapted from https://github.com/lorenzMuller/kernelNet_MovieLens/blob/master/kernelNet_ml1m.py
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from easydict import EasyDict as edict

from lib.models.base_model import BaseModel

params = edict()
params.HIDDEN_UNITS = 500
params.TESTING = False
params.LAMBDA_2 = 90.
params.LAMBDA_SPARSITY = 0.023
params.N_LAYERS = 2
params.MAX_ITER = 50 if not params.TESTING else 5  # evaluate performance on test set; breaks l-bfgs loop
params.N_EPOCHS = 3 if not params.TESTING else 3
params.VERBOSE_DISP = False


class KernelNet(BaseModel):
    def __init__(self, config, logger, model_nr):
        super().__init__(logger, model_nr)
        self.config = config
        self.R = tf.placeholder("float", [None, self.config.NUM_USERS])

        y = self.R
        reg_losses = None
        for i in range(params.N_LAYERS):
            y, reg_loss = self.kernel_layer(y, params.HIDDEN_UNITS, name=str(i), lambda_2=params.LAMBDA_2,
                                            lambda_s=params.LAMBDA_SPARSITY)
            reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss
        self.prediction, reg_loss = self.kernel_layer(y, self.config.NUM_USERS, activation=tf.identity, name='out',
                                                      lambda_2=params.LAMBDA_2, lambda_s=params.LAMBDA_SPARSITY)
        self.reg_losses = reg_losses + reg_loss

    def fit_init(self, train_mask):
        diff = train_mask * (self.R - self.prediction)
        sqE = tf.nn.l2_loss(diff)
        loss = sqE + self.reg_losses

        # Instantiate L-BFGS Optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',
                                                                options={'maxiter': params.MAX_ITER,
                                                                         'disp': params.VERBOSE_DISP,
                                                                         'maxcor': 10},
                                                                )

    def kernel_layer(self, x, hidden_units=500, n_dim=5, activation=tf.nn.sigmoid, lambda_s=0.013,
                     lambda_2=60., name=''):
        with tf.variable_scope(name):
            W = tf.get_variable('W', [x.shape[1], hidden_units])
            n_in = x.get_shape().as_list()[1]
            u = tf.get_variable('u', initializer=tf.random.truncated_normal([n_in, 1, n_dim], 0., 1e-3))
            v = tf.get_variable('v', initializer=tf.random.truncated_normal([1, hidden_units, n_dim], 0., 1e-3))
            b = tf.get_variable('b', [hidden_units])

        # kernel
        dist = tf.norm(u - v, ord=2, axis=2)
        w_hat = tf.maximum(0., 1. - dist ** 2)

        # regularization
        sparse_reg = tf.contrib.layers.l2_regularizer(lambda_s)
        sparse_reg_term = tf.contrib.layers.apply_regularization(sparse_reg, [w_hat])

        l2_reg = tf.contrib.layers.l2_regularizer(lambda_2)
        l2_reg_term = tf.contrib.layers.apply_regularization(l2_reg, [W])

        W_eff = W * w_hat
        y = tf.matmul(x, W_eff) + b
        y = activation(y)
        return y, sparse_reg_term + l2_reg_term

    def fit(self, train_data, train_predictions, **kwargs):
        val_data, val_predictions = self.get_kwargs_data(kwargs, 'val_data', 'val_predictions')
        test_data, test_every = self.get_kwargs_data(kwargs, 'test_data', 'test_every')

        data, mask = self.create_matrices(train_data['movie_id'].values, train_data['user_id'].values,
                                          train_predictions['rating'].values,
                                          default_replace=self.config.DEFAULT_VALUES[self.model_nr])
        data, mask = data.T, mask.T  # NOTE transpose
        if not val_data is None:
            data_val, mask_val = self.create_matrices(val_data['movie_id'].values, val_data['user_id'].values,
                                                      val_predictions,
                                                      default_replace=self.config.DEFAULT_VALUES[self.model_nr])
            data_val, mask_val = data_val.T, mask_val.T  # NOTE transpose

        self.fit_init(mask)

        errors = pd.DataFrame({'train_error': [], 'valid_error': []})

        # Training and validation loop
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(params.N_EPOCHS):
                self.optimizer.minimize(sess, feed_dict={self.R: data})  # do maxiter optimization steps
                pre = sess.run(self.prediction, feed_dict={self.R: data})  # predict ratings

                error_val = 0 if val_data is None \
                    else np.sqrt((mask_val * (np.clip(pre, 1., 5.) - data_val) ** 2).sum() / mask_val.sum())
                error_train = np.sqrt((mask * (np.clip(pre, 1., 5.) - data) ** 2).sum() / mask.sum())
                if i % self.config.TEST_EVERY == 0:
                    self.validation_rmse.append(np.round(np.sqrt(error_val), 4).item())

                errors = errors.append({'train_error': error_train, 'valid_error': error_val}, ignore_index=True)
                errors.to_csv(self.config.FINAL_OUTPUT_DIR + '/errors.csv')

                self.log_info(
                    f'epoch: {i}, validation rmse: {np.round(error_val, 4)}, train rmse: {np.round(error_train, 4)}')

                self.reconstructed_matrix = pre

                if i == 0 and params.TESTING:
                    break

                if not test_data is None and (i + 1) % test_every == 0:
                    self.log_info(f'Creating submission for epoch {i} with train_err {error_train}')
                    self.create_submission(test_data, suffix=f'_e{i}_err{error_train:.2f}')

    def predict(self, test_data):
        predictions, self.index = self._extract_prediction_from_full_matrix(self.reconstructed_matrix.transpose(),
                                                                            users=test_data['user_id'].values,
                                                                            movies=test_data['movie_id'].values)
        predictions = self.postprocessing(predictions)
        return predictions

    def create_submission(self, X, suffix='', postprocessing='clipping'):
        predictions = self.postprocessing(self.predict(X), postprocessing)
        self.save_submission(self.index, predictions, suffix=suffix)
        return predictions


def get_model(config, logger, model_nr=0):
    return KernelNet(config, logger, model_nr)
