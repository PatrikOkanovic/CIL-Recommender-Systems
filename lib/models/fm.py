'''
Adapted from https://github.com/tohtsky/myFM/blob/master/examples/ml-1m-extended.ipynb
'''
import os.path
from collections import defaultdict

import hickle as hkl
import myfm
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from myfm import RelationBlock
from scipy import sparse
from tqdm import tqdm

from lib.models.base_model import BaseModel

params = edict()
params.FEATURES_PATH = 'data/features/'
params.RANK = 12  # rank 12 is best
params.N_ITER = 512  # does not work if not enough iterations
params.SAMPLES = None  # default is params.N_ITER - 5
params.GROUPING = True
params.USE_IU = True  # use implicit user feature (SVD++)
params.USE_II = True  # use implicit item feature (SVD++ flipped)
params.USE_JACCARD = False  # taken from "Improving Jaccard Index for Measuring Similarity in Collaborative Filtering"
params.USE_JACCARDPP = params.USE_JACCARD and False  # Taken from "Improving Jaccard Index for Measuring Similarity in Collaborative Filtering", using L=3 and H=4
params.USE_MOVIE = False
params.USE_DISTANCES = 'euclidean'  # can be either '', 'euclidean' or 'mahalanobis'
params.USE_GENRES = not params.USE_DISTANCES
params.ORDERED_PROBIT = True  # must be set to true for reproducing the best model


class BFM(BaseModel):
    def __init__(self, config, logger, model_nr=0, rank=params.RANK, samples=params.SAMPLES,
                 use_iu=params.USE_IU,
                 use_ii=params.USE_II):
        super().__init__(logger, model_nr)
        self.config = config
        params.RANK = rank
        params.SAMPLES = samples
        params.USE_IU = use_iu
        params.USE_II = use_ii

        self.num_users = self.config.NUM_USERS
        self.num_movies = self.config.NUM_MOVIES
        self._load_feature_data(logger)

    def set_params(self, logger, config, rank, samples, use_iu, use_ii):
        self.logger = logger
        self.config = config
        params.SAMPLES = samples
        params.RANK = rank
        params.USE_IU = use_iu
        params.USE_II = use_ii
        return self

    def get_params(self, deep):
        return {"samples": params.SAMPLES, "rank": params.RANK, "use_iu": params.USE_IU, "use_ii": params.USE_II,
                "logger": self.logger, "config": self.config}

    def _load_feature_data(self, logger):
        if params.USE_JACCARD:
            if params.USE_JACCARDPP \
                    and os.path.isfile(params.FEATURES_PATH + 'jaccard_L_gzip.hkl') \
                    and os.path.isfile(params.FEATURES_PATH + 'jaccard_H_gzip.hkl'):
                logger.info('Loading dumped Jaccard matrix')
                L = hkl.load(params.FEATURES_PATH + 'jaccard_L_gzip.hkl')
                H = hkl.load(params.FEATURES_PATH + 'jaccard_H_gzip.hkl')
                self.jaccard = (L + H) / 2
            elif os.path.isfile(params.FEATURES_PATH + 'jaccard_gzip.hkl'):
                logger.info('Loading dumped Jaccard matrix')
                self.jaccard = hkl.load(params.FEATURES_PATH + 'jaccard_gzip.hkl')
            else:
                logger.info('Jaccard matrix not found, recalculating values')
                self.jaccard = None
        if params.USE_MOVIE:
            if params.USE_DISTANCES and os.path.isfile(params.FEATURES_PATH + f'{params.USE_DISTANCES}_matrix.npy'):
                logger.info(f'Loading dumped {params.USE_DISTANCES} distance matrix')
                self.movie_features = np.load(params.FEATURES_PATH + f'{params.USE_DISTANCES}_matrix.npy')
            elif params.USE_GENRES and os.path.isfile(params.FEATURES_PATH + 'rank18_movie_categories.npy'):
                logger.info(f'Loading dumped movie genres matrix')
                self.movie_features = np.load(params.FEATURES_PATH + 'rank18_movie_categories.npy')
            else:
                logger.info('Movie feature matrix not found, either create matrix or turn off feature')
                logger.info('Shutting down...')
                exit(0)

    def fit(self, train_data, train_predictions, **kwargs):
        self.df_train = pd.concat([train_data, train_predictions], axis=1)

        self.unique_user_ids = np.unique(self.df_train.user_id)
        self.unique_movie_ids = np.unique(self.df_train.movie_id)
        self.user_id_to_index = {uid: i for i, uid in enumerate(self.unique_user_ids)}
        self.movie_id_to_index = {mid: i for i, mid in enumerate(self.unique_movie_ids)}

        self.movie_vs_watched = dict()
        self.user_vs_watched = dict()
        self.user_vs_watched_L = defaultdict(list)  # L=3
        self.user_vs_watched_H = defaultdict(list)  # H=4
        for row in self.df_train.itertuples():
            user_id = row.user_id
            movie_id = row.movie_id
            rating = row.rating
            self.movie_vs_watched.setdefault(movie_id, list()).append(user_id)
            self.user_vs_watched.setdefault(user_id, list()).append(movie_id)
            if rating <= 3:
                self.user_vs_watched_L.setdefault(user_id, list()).append(movie_id)
            else:
                self.user_vs_watched_H.setdefault(user_id, list()).append(movie_id)

        # Create RelationBlock.
        train_blocks = self._create_relational_blocks(self.df_train)

        # setup grouping, which specifies each variable group’s size.
        group_shapes = []
        self._set_group_shapes(group_shapes)

        if not params.ORDERED_PROBIT:
            self.fm = myfm.MyFMRegressor(rank=params.RANK)
            self.fm.fit(
                None, self.df_train.rating.values, X_rel=train_blocks,
                group_shapes=group_shapes,
                n_iter=params.N_ITER,
                n_kept_samples=params.SAMPLES
            )
        else:
            self.fm = myfm.MyFMOrderedProbit(rank=params.RANK)
            # rating values are set to [0, 1, 2, 3, 4]
            self.fm.fit(
                None, self.df_train.rating.values - 1, train_blocks,
                group_shapes=group_shapes,
                n_iter=params.N_ITER,
                n_kept_samples=None
            )

    def _set_group_shapes(self, group_shapes):
        group_shapes.append(len(self.user_id_to_index))  # user ids
        if params.USE_IU:
            group_shapes.append(len(self.movie_id_to_index))
        if params.USE_JACCARD:
            group_shapes.append(len(self.user_id_to_index))
        group_shapes.append(len(self.movie_id_to_index))  # movie ids
        if params.USE_II:
            group_shapes.append(len(self.user_id_to_index))  # all users who watched the movies
        if params.USE_MOVIE:
            if params.USE_GENRES:
                group_shapes.append(18)  # rank/number of movie genres
            elif params.USE_DISTANCES:
                group_shapes.append(len(self.movie_id_to_index))

    def predict(self, test_data):
        self.df_test = test_data

        print('Starting prediction')
        if not params.ORDERED_PROBIT:
            # Create RelationBlock.
            test_blocks = self._create_relational_blocks(self.df_test)
            print('Starting prediction')
            predictions = self.fm.predict(None, test_blocks)
        else:
            print('Starting prediction')
            n = 10000  # split data frame in chunks of size 10000 to require less memory
            list_df = [self.df_test[i:i + n] for i in range(0, self.df_test.shape[0], n)]
            predictions = []
            for chunk in tqdm(list_df):
                unique_users, user_map = np.unique(chunk.user_id, return_inverse=True)
                unique_movies, movie_map = np.unique(chunk.movie_id, return_inverse=True)
                user_data = self._augment_user_id(unique_users, self.user_id_to_index, self.movie_id_to_index,
                                                  self.user_vs_watched)[user_map]
                movie_data = self._augment_movie_id(unique_movies, self.movie_id_to_index, self.user_id_to_index,
                                                    self.movie_vs_watched)[movie_map]
                X_test = sparse.hstack([user_data, movie_data])

                prediction = self.fm.predict_proba(X_test)
                # Calculate expected value over class probabilities. Rating values are now [1, 2, 3, 4, 5]
                predictions.append(prediction.dot(np.arange(1, 6)))
            predictions = np.hstack(predictions)

        predictions = self.postprocessing(predictions)

        print('Finishing prediction')
        return predictions

    def _create_relational_blocks(self, df):
        blocks = []
        for source, target in [(df, blocks)]:
            unique_users, user_map = np.unique(source.user_id, return_inverse=True)
            target.append(
                RelationBlock(user_map,
                              self._augment_user_id(unique_users, self.user_id_to_index, self.movie_id_to_index,
                                                    self.user_vs_watched).tocsr())
            )
            unique_movies, movie_map = np.unique(source.movie_id, return_inverse=True)
            target.append(
                RelationBlock(movie_map,
                              self._augment_movie_id(unique_movies, self.movie_id_to_index, self.user_id_to_index,
                                                     self.movie_vs_watched).tocsr())
            )

        return blocks

    def _augment_user_id(self, user_ids, user_id_to_index, movie_id_to_index, user_vs_watched):
        X = sparse.lil_matrix((len(user_ids), len(user_id_to_index) + (len(movie_id_to_index) if params.USE_IU else 0) +
                               (len(user_id_to_index) if params.USE_JACCARD else 0)))
        # index and user_id are equal
        for index, user_id in enumerate(tqdm(user_ids)):
            if user_id in user_id_to_index:
                X[index, user_id_to_index[user_id]] = 1
            if params.USE_IU:
                watched_movies = user_vs_watched.get(user_id, [])
                normalizer = 1 / max(len(watched_movies), 1) ** 0.5
                for mid in watched_movies:
                    if mid in movie_id_to_index:
                        X[index, len(user_id_to_index) + movie_id_to_index[mid]] = normalizer
            if params.USE_JACCARD:
                for uid in user_ids:
                    if self.jaccard is not None:
                        X[index, len(user_id_to_index) + len(movie_id_to_index) + user_id_to_index[uid]] = \
                            self.jaccard[user_id, uid]
                    else:
                        X[index, len(user_id_to_index) + len(movie_id_to_index) + user_id_to_index[uid]] = \
                            jaccard(user_id, uid, user_vs_watched) \
                                if not params.USE_JACCARDPP \
                                else (jaccard(user_id, uid, self.user_vs_watched_L) + jaccard(user_id, uid,
                                                                                              self.user_vs_watched_H)) / 2

        return X

    def _augment_movie_id(self, movie_ids, movie_id_to_index, user_id_to_index, movie_vs_watched):
        X = sparse.lil_matrix(
            (len(movie_ids), len(movie_id_to_index) + (len(user_id_to_index) if params.USE_II else 0)
             + (len(movie_id_to_index) if params.USE_MOVIE and params.USE_DISTANCES else 0)
             + (18 if params.USE_MOVIE and params.USE_GENRES else 0)))
        # index and movie_id are equal
        for index, movie_id in enumerate(tqdm(movie_ids)):
            if movie_id in movie_id_to_index:
                X[index, movie_id_to_index[movie_id]] = 1
            if params.USE_II:
                watched_users = movie_vs_watched.get(movie_id, [])
                normalizer = 1 / max(len(watched_users), 1) ** 0.5
                for uid in watched_users:
                    if uid in user_id_to_index:
                        X[index, len(movie_id_to_index) + user_id_to_index[uid]] = normalizer
            if params.USE_MOVIE:
                if params.USE_DISTANCES:
                    for mid in movie_ids:
                        X[index, len(movie_id_to_index) + len(user_id_to_index) + mid] = self.movie_features[index, mid]
                elif params.USE_GENRES:
                    X[index, len(movie_id_to_index) + len(user_id_to_index) + self.movie_features[index]] = 1
        return X

    def create_submission(self, X, suffix='', postprocessing='clipping'):
        predictions = self.postprocessing(self.predict(X), postprocessing)
        index = [''] * X.shape[0]
        for i, row in enumerate(X.itertuples()):
            index[i] = f"r{row.user_id + 1}_c{row.movie_id + 1}"
        submission = pd.DataFrame({'Id': index, 'Prediction': predictions})
        submission.to_csv(self.config.SUBMISSION_NAME, index=False)
        return predictions


def jaccard(u, v, user_vs_watched):
    set_u = set(user_vs_watched[u])
    set_v = set(user_vs_watched[v])
    if len(set_u) == 0 and len(set_v) == 0:
        return 0
    return len(set_u & set_v) / len(set_u | set_v)


def get_model(config, logger, model_nr=0):
    return BFM(config, logger, model_nr)
