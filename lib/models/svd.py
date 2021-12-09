import numpy as np
from easydict import EasyDict as edict

from lib.models.base_model import BaseModel
from lib.utils.config import config

params = edict()
params.RANK = 3


class SVD(BaseModel):
    def __init__(self, config, logger, model_nr=0, rank=params.RANK):
        super().__init__(logger, model_nr)
        self.config = config
        self.num_users = config.NUM_USERS
        self.num_movies = config.NUM_MOVIES
        params.RANK = rank

    def set_params(self, logger, config, rank):
        self.logger = logger
        self.config = config
        params.RANK = rank
        return self

    def get_params(self, deep):
        return {"rank": params.RANK, "logger": self.logger, "config": self.config}

    def fit(self, train_data, train_predictions, **kwargs):
        # create full matrix of observed and unobserved values
        data, mask = self.create_matrices(train_data['movie_id'].values, train_data['user_id'].values,
                                          train_predictions['rating'].values,
                                          default_replace=config.DEFAULT_VALUES[self.model_nr])
        number_of_singular_values = min(self.num_users, self.num_movies)
        assert (params.RANK <= number_of_singular_values), "choose correct number of singular values"
        U, s, Vt = np.linalg.svd(data, full_matrices=False)
        S = np.zeros((self.num_movies, self.num_movies))
        S[:params.RANK, :params.RANK] = np.diag(s[:params.RANK])
        self.reconstructed_matrix = U.dot(S).dot(Vt)

    def predict(self, test_data):
        assert (len(test_data['user_id']) == len(
            test_data['movie_id'])), "users-movies combinations specified should have equal length"
        predictions, self.index = self._extract_prediction_from_full_matrix(self.reconstructed_matrix,
                                                                            users=test_data['user_id'].values,
                                                                            movies=test_data['movie_id'].values)
        predictions = self.postprocessing(predictions, 'clipping')
        return predictions

    def create_submission(self, test_data, suffix='', postprocessing='clipping'):
        predictions = self.postprocessing(self.predict(test_data), postprocessing)
        self.save_submission(self.index, predictions, suffix)
        return predictions


def get_model(config, logger, model_nr=0):
    return SVD(config, logger, model_nr)
