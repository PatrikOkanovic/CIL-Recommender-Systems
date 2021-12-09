from easydict import EasyDict as edict
from sklearn.decomposition import non_negative_factorization

from lib.models.base_model import BaseModel
from lib.utils.config import config

params = edict()
params.RANK = 24
params.INIT_PROCEDURE = 'nndsvda'
params.MAX_ITER = 512


class NMF(BaseModel):
    def __init__(self, config, logger, model_nr=0, rank=params.RANK):
        super().__init__(logger, model_nr)
        self.config = config
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
        W, H, _ = non_negative_factorization(data, n_components=params.RANK, init=params.INIT_PROCEDURE, verbose=True,
                                             max_iter=params.MAX_ITER)
        self.reconstructed_matrix = W @ H

    def predict(self, test_data):
        predictions, self.index = self._extract_prediction_from_full_matrix(self.reconstructed_matrix,
                                                                            users=test_data['user_id'].values,
                                                                            movies=test_data['movie_id'].values)
        predictions = self.postprocessing(predictions)
        return predictions

    def create_submission(self, test_data, suffix='', postprocessing='clipping'):
        predictions = self.postprocessing(self.predict(test_data), postprocessing)
        self.save_submission(self.index, predictions, suffix)
        return predictions


def get_model(config, logger, model_nr=0):
    return NMF(config, logger, model_nr)
