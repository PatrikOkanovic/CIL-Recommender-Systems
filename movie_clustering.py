import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

import lib.models as models
from lib.utils import utils
from lib.utils.config import config
from lib.utils.loader import extract_users_items_predictions, read_data
from movie_distances import get_embeddings


def perform_kmeans(embeddings, embeddings_2d):
    # MovieLens 18 categories + no_category
    for num_cluster in range(5, 30):
        kmeans = KMeans(n_clusters=num_cluster).fit(embeddings)
        plt.figure()
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=kmeans.labels_, s=40, cmap='viridis')
        plt.title("Clustering {} movie categories".format(num_cluster))
        plt.savefig('clustering{}.png'.format(num_cluster))


def plot_histogram(labels, bins):
    values = np.bincount(labels)
    categories = [str(int) for int in range(bins)]
    plt.bar(categories, values)
    plt.title('Movie category distribution')
    plt.xlabel('Movie category')
    plt.ylabel('Number of movies')
    plt.savefig('movie_category_distribution.png')


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
    embeddings_2d = get_embeddings(data, rank=2)
    perform_kmeans(embeddings, embeddings_2d)

    kmeans = KMeans(n_clusters=18).fit(embeddings)
    rank18_labels = kmeans.labels_
    np.save('rank18_movie_categories.npy', rank18_labels)
    plot_histogram(rank18_labels, 18)
