import os

import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.models import models
from lib.models.autoencoder import AutoEncoder
from lib.utils import utils
from lib.utils.config import config
from lib.utils.loader import extract_users_items_predictions, read_data
from lib.utils.utils import get_score


def plot_deep_autoencoder(encoded_dimension, rmse, path):
    plt.plot(encoded_dimension, rmse[0], '-x', c='b', label='1 layer')
    plt.plot(encoded_dimension, rmse[1], '-D', c='r', label='2 layers')

    plt.xlabel('Encoded dimension')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(path)


def call_deep_autoencoder(X_train, y_train, X_val, val_predictions):
    encoded_dimensions = [16, 32, 50, 100, 250, 350, 500]
    one_layer_rmse = []
    two_layer_rmse = []

    for encoded_dimension in encoded_dimensions:
        model = AutoEncoder(config, logger, encoded_dimension=encoded_dimension, single_layer=True)
        model.fit(X_train, y_train,
                  val_data=X_val, val_predictions=val_predictions)  # iterative val score
        logger.info("Testing the model")
        predictions = model.predict(X_val)
        rmse = get_score(predictions, target_values=val_predictions)
        one_layer_rmse.append(rmse)

    for encoded_dimension in encoded_dimensions:
        model = AutoEncoder(config, logger, encoded_dimension=encoded_dimension, single_layer=False,
                            hidden_dimension=[500])
        model.fit(X_train, y_train,
                  val_data=X_val, val_predictions=val_predictions)  # iterative val score
        logger.info("Testing the model")
        predictions = model.predict(X_val)
        rmse = get_score(predictions, target_values=val_predictions)
        two_layer_rmse.append(rmse)

    plot_deep_autoencoder(encoded_dimensions, [one_layer_rmse, two_layer_rmse], 'deep_autoencoder.png')


def plot_heatmap(matrix, x_values, y_values, show_values=False):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap='RdBu')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(x_values)),
           yticks=np.arange(len(y_values)),
           xticklabels=x_values, yticklabels=y_values,
           title="Heatmap",
           ylabel='Sample size',
           xlabel='Rank')

    if show_values:
        fmt = '.2f'
        thresh = matrix.max() / 2.
        for i in range(len(x_values)):
            for j in range(len(y_values)):
                ax.text(j, i, format(matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if matrix[i, j] > thresh else "black")

    fig.savefig('heatmap.png')


def plot_rmse(rank, rmse_values, markers, colors, labels, path, x_title, y_title):
    plt.figure()
    for i in range(len(rmse_values)):
        plt.plot(rank, rmse_values[i], markers[i], c=colors[i], label=labels[i])
    plt.legend()
    plt.xlabel(x_title)
    if x_title == 'Rank':
        plt.xticks(rank)
    plt.ylabel(y_title)
    plt.savefig(path)


def call_rmse_validation(X_train, y_train, X_val, val_predictions):
    model_names = ['autoencoder', 'autorec', 'ncf', 'kernel_net']
    rmse_values = []
    for model_name in model_names:
        if os.path.exists(model_name + '.npy'):
            data = np.load(model_name + '.npy')
            rmse_values.append(data.tolist()[0:25])
        else:
            # Set the config
            config.MODEL = model_name
            model = models[config.MODEL].get_model(config, logger)
            model.fit(X_train, y_train,
                      val_data=X_val, val_predictions=val_predictions)
            np.save(model_name + '.npy', np.array(model.validation_rmse))
            rmse_values.append(model.validation_rmse)
    np.save('validation_rmse.npy', np.array(rmse_values))
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['-x', '-d', '-h', '-s']
    plot_rmse(range(0, 125, 5), rmse_values, markers, colors,
              ['Autoencoder', 'AutoRec', 'NCF', 'KernelNet'], 'validation_plot.png', 'Epoch', 'Validation RMSE')


def call_rmse_rank():
    rmse_values = []
    svd = hkl.load('svd.hkl')
    rmse_values.append(svd['mean_test_score'].tolist())
    nmf = hkl.load('nmf.hkl')
    rmse_values.append(nmf['mean_test_score'].tolist())
    bfm = hkl.load('bfm.hkl')
    rmse_values.append(bfm['mean_test_score'].tolist())
    bfm_svdpp = hkl.load('bfm_svdpp.hkl')
    rmse_values.append(bfm_svdpp['mean_test_score'].tolist())
    bfm_svdpp_flipped = hkl.load('bfm_svdpp_flipped.hkl')
    rmse_values.append(bfm_svdpp_flipped['mean_test_score'].tolist()[4::5])

    ranks = [dict['rank'] for dict in svd['params']]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['-x', '-d', '-h', '-s', '-*']
    labels = ['SVD', 'NMF', 'BFM', 'BFM SVD++', 'BFM SVD++ flipped']

    plot_rmse(ranks, rmse_values, markers, colors, labels, 'rank.png', 'Rank', 'Validation RMSE')


def call_heatmap():
    bfm_svdpp_flipped = hkl.load('bfm_svdpp_flipped.hkl')
    ranks = []
    samples = []
    for dict in bfm_svdpp_flipped['params']:
        ranks.append(dict['rank'])
        samples.append(dict['samples'])
    ranks = np.unique(ranks).tolist()
    samples = np.unique(samples).tolist()
    heatmap = np.zeros((len(samples), len(ranks)))
    for i, dict in enumerate(bfm_svdpp_flipped['params']):
        heatmap[samples.index(dict['samples']), ranks.index(dict['rank'])] = bfm_svdpp_flipped['mean_test_score'][i]

    plot_heatmap(heatmap, ranks, samples)


if __name__ == '__main__':
    assert config.TYPE == 'VAL', 'config.TYPE has to be VAL'
    logger = utils.init(seed=config.RANDOM_STATE)
    train_pd, val_pd, test_pd = read_data()
    train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
    val_users, val_movies, val_predictions = extract_users_items_predictions(val_pd)
    test_users, test_movies, _ = extract_users_items_predictions(test_pd)

    X_train = pd.DataFrame({'user_id': train_users, 'movie_id': train_movies})
    y_train = pd.DataFrame({'rating': train_predictions})
    X_val = pd.DataFrame({'user_id': val_users, 'movie_id': val_movies})
    X_test = pd.DataFrame({'user_id': test_users, 'movie_id': test_movies})

    call_rmse_rank()
    call_rmse_validation(X_train, y_train, X_val, val_predictions)
    call_heatmap()
