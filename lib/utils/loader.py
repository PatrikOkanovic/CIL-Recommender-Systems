import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lib.utils.config import config


def read_data():
    data_directory = config.DATA_DIR
    data_pd = pd.read_csv(f'{data_directory}/data_train.csv')
    movies, users, predictions = extract_users_items_predictions(data_pd)
    data_pd['users'], data_pd['movies'] = users, movies
    print(data_pd.head(5))
    print()
    print('Shape', data_pd.shape)

    test_pd = pd.read_csv(f'{data_directory}/sampleSubmission.csv')

    if config.TYPE == 'VAL':
        train_pd, val_pd = train_test_split(data_pd, train_size=config.TRAIN_SIZE, random_state=config.RANDOM_STATE,
                                            stratify=data_pd[config.STRATIFY])
        return train_pd, val_pd, test_pd
    else:
        return data_pd, test_pd


def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in
         np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    # index for users and movies starts at 0
    return users, movies, predictions
