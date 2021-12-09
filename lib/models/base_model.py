from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from lib import models
from lib.utils.config import config
from lib.utils.utils import roundPartial


class BaseModel(ABC):
    def __init__(self, logger, model_nr=0):
        self.logger = logger
        self.model_nr = model_nr
        self.validation_rmse = []

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def create_submission(self, X, suffix='', postprocessing='clipping'):
        pass

    def get_kwargs_data(self, kwargs, *keys):
        output = []
        for k in keys:
            output.append(kwargs[k] if k in kwargs else None)

        return output

    # necessary when instantiating model with no logger e.g. for initializing unobserved values
    def log_info(self, msg):
        if self.logger != None:
            self.logger.info(msg)
        else:
            print(msg)

    def _extract_prediction_from_full_matrix(self, reconstructed_matrix, users, movies):
        # returns predictions for the users-movies combinations specified based on a full m \times n matrix
        predictions = np.zeros(len(users))
        index = [''] * len(users)

        for i, (user, movie) in enumerate(zip(users, movies)):
            predictions[i] = reconstructed_matrix[user][movie]
            index[i] = f"r{user + 1}_c{movie + 1}"

        return predictions, index

    def save_submission(self, index, predictions, suffix=''):
        submission = pd.DataFrame({'Id': index, 'Prediction': predictions})
        filename = config.SUBMISSION_NAME
        if suffix != '':
            filename = config.SUBMISSION_NAME[:-4] + suffix + '.csv'
        submission.to_csv(filename, index=False)

    def create_matrices(self, train_movies, train_users, train_predictions, default_replace='mean'):
        data = np.full((config.NUM_USERS, config.NUM_MOVIES), 0, dtype=float)
        mask = np.zeros((config.NUM_USERS, config.NUM_MOVIES))  # 0 -> unobserved value, 1->observed value
        for user, movie, pred in zip(train_users, train_movies, train_predictions):
            data[user][movie] = pred
            mask[user][movie] = 1

        mean = np.mean(train_predictions)

        if default_replace == 'zero':
            pass
        elif default_replace == 'mean':
            data[mask == 0] = mean
        elif default_replace == 'user_mean':
            for i in range(0, config.NUM_USERS):
                data[i, mask[i, :] == 0] = mean if len(data[i, :][mask[i, :] == 1]) == 0 else np.mean(
                    data[i, :][mask[i, :] == 1])
        elif default_replace == 'item_mean':
            for i in range(0, config.NUM_MOVIES):
                data[mask[:, i] == 0, i] = mean if len(data[:, i][mask[:, i] == 1]) == 0 else np.mean(
                    data[:, i][mask[:, i] == 1])
        else:
            if default_replace not in models.models:
                raise NotImplementedError('Add other replacement methods')

            unobserved_initializer_model = models.models[default_replace].get_model(config, logger=self.logger,
                                                                                    model_nr=self.model_nr + 1)

            data = self.use_model_to_init_unobserved(data, mask,
                                                     unobserved_initializer_model,
                                                     train_movies, train_predictions, train_users)
        self.log_info(f'Used {default_replace} to initialize unobserved entries as step {self.model_nr}')

        return data, mask

    def use_model_to_init_unobserved(self, data, mask, unobserved_initializer_model, train_movies, train_predictions,
                                     train_users):

        X_train = pd.DataFrame({'user_id': train_users, 'movie_id': train_movies})
        y_train = pd.DataFrame({'rating': train_predictions})
        unobserved_initializer_model.fit(X_train, y_train)
        unobserved_indices = np.argwhere(mask == 0)
        unobserved_users, unobserved_movies = [unobserved_indices[:, c] for c in [0, 1]]
        X_test = pd.DataFrame({'user_id': unobserved_users, 'movie_id': unobserved_movies})
        predictions = unobserved_initializer_model.predict(X_test)
        for i in range(len(unobserved_indices)):
            user, movie = unobserved_indices[i]
            data[user][movie] = predictions[i]

        return data

    def postprocessing(self, predictions, type='clipping'):
        if type == 'round_quarters':
            predictions = roundPartial(predictions, 0.25)
            print(f'Used {type} for postprocessing')
        elif type == 'clipping':
            print(f'Used {type} for postprocessing')
        elif type == 'round':
            predictions = roundPartial(predictions, 1)
            print(f'Used {type} for postprocessing')
        else:
            print(f'Invalid preprocessing step: {type}')
        predictions = np.clip(predictions, 1., 5.)
        return predictions
