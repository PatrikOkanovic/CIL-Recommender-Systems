import hickle as hkl
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV

from lib.models.fm import BFM
from lib.models.nmf import NMF
from lib.models.svd import SVD
from lib.utils import utils
from lib.utils.config import config
from lib.utils.loader import extract_users_items_predictions, read_data


def main():
    logger = utils.init(seed=config.RANDOM_STATE)
    config.TYPE = 'ALL'
    train_pd, _ = read_data()
    train_pd = train_pd.sample(frac=1)
    train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
    X = pd.DataFrame({'user_id': train_users, 'movie_id': train_movies})
    y = pd.DataFrame({'rating': train_predictions})

    parameters = {"samples": [32, 64, 128, 256, 512], "rank": [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
                  "use_iu": [True], "use_ii": [True], 'logger': [logger], 'config': [config]}
    clf = GridSearchCV(BFM(config, logger), parameters, scoring=make_scorer(mean_squared_error, squared=False), cv=3,
                       n_jobs=-1, verbose=1)

    clf.fit(X, y)
    prepare_save_file(clf.cv_results_, 'bfm_svdpp_flipped')

    parameters = {"samples": [512], "rank": [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
                  "use_iu": [True], "use_ii": [False], 'logger': [logger], 'config': [config]}
    clf = GridSearchCV(BFM(config, logger), parameters, scoring=make_scorer(mean_squared_error, squared=False), cv=3,
                       n_jobs=-1, verbose=1)

    clf.fit(X, y)
    prepare_save_file(clf.cv_results_, 'bfm_svdpp')

    parameters = {"samples": [512], "rank": [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
                  "use_iu": [False], "use_ii": [False], 'logger': [logger], 'config': [config]}
    clf = GridSearchCV(BFM(config, logger), parameters, scoring=make_scorer(mean_squared_error, squared=False), cv=3,
                       n_jobs=-1, verbose=1)

    clf.fit(X, y)
    prepare_save_file(clf.cv_results_, 'bfm')

    parameters = {"rank": [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
                  'logger': [logger], 'config': [config]}
    clf = GridSearchCV(SVD(config, logger), parameters, scoring=make_scorer(mean_squared_error, squared=False), cv=3,
                       n_jobs=-1, verbose=1)

    clf.fit(X, y)
    prepare_save_file(clf.cv_results_, 'svd')

    parameters = {"rank": [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
                  'logger': [logger], 'config': [config]}
    clf = GridSearchCV(NMF(config, logger), parameters, scoring=make_scorer(mean_squared_error, squared=False), cv=3,
                       n_jobs=-1, verbose=1)

    clf.fit(X, y)
    prepare_save_file(clf.cv_results_, 'nmf')


def prepare_save_file(cv_results, name):
    params_to_drop = []
    for k in cv_results.keys():
        if k.startswith('param_'):
            params_to_drop.append(k)
    for k in params_to_drop:
        cv_results.pop(k)
    for i in range(len(cv_results['params'])):
        cv_results['params'][i].pop('config')
        cv_results['params'][i].pop('logger')
    hkl.dump(cv_results, f'{name}.hkl', mode='w')


if __name__ == '__main__':
    main()
