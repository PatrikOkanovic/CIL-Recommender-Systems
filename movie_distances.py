import os

import numpy as np
from scipy.spatial import distance

import lib.models as models
from lib.utils import utils
from lib.utils.config import config
from lib.utils.loader import extract_users_items_predictions, read_data


def get_embeddings(data, rank=10):
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    S = np.zeros((data.shape[0], data.shape[1]))
    S[:data.shape[1], :data.shape[1]] = np.diag(s)
    embeddings = np.sqrt(S).dot(Vt)[:rank, :]
    return embeddings.T


def get_distances(embeddings):
    covariance = np.cov(embeddings.T)
    inv_covariance = np.linalg.inv(covariance)
    euclidean_distances = {}
    mahalanobis_distances = {}
    for i in range(1000):
        for j in range(i, 1000):
            euclidean_distances[(i, j)] = np.linalg.norm(embeddings[i] - embeddings[j])
            mahalanobis_distances[(i, j)] = distance.mahalanobis(embeddings[i], embeddings[j], inv_covariance)

    euclidean_matrix = np.zeros((embeddings.shape[0], embeddings.shape[0]))
    mahalanobis_matrix = np.zeros((embeddings.shape[0], embeddings.shape[0]))
    for i in range(1000):
        for j in range(1000):
            euclidean_matrix[i, j] = euclidean_distances[(min(i, j), max(i, j))]
            mahalanobis_matrix[i, j] = mahalanobis_distances[(min(i, j), max(i, j))]

    return euclidean_matrix, mahalanobis_matrix


if __name__ == '__main__':
    if os.path.exists('data.npy'):
        data = np.load('data.npy')
    else:
        logger = utils.init(seed=config.RANDOM_STATE)
        logger.info(f'Using {config.MODEL} model for prediction')
        # Load data
        config.TYPE = 'ALL'
        data_pd, test_pd = read_data()
        users, movies, predictions = extract_users_items_predictions(data_pd)
        model = models.models[config.MODEL].get_model(config, logger)
        data, mask = model.create_matrices(movies, users, predictions, config.DEFAULT_VALUE)
        np.save('data.npy', data)

    embeddings = get_embeddings(data)
    euclidean_matrix, mahalanobis_matrix = get_distances(embeddings)
    np.save('euclidean_matrix.npy', euclidean_matrix)
    np.save('mahalanobis_matrix.npy', mahalanobis_matrix)
