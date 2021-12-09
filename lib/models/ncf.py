import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from lib.models.base_model import BaseModel
from lib.utils.utils import get_score

params = edict()
params.EMBEDDING_SIZE = 16
params.LEARNING_RATE = 0.001
params.BATCH_SIZE = 64
params.NUM_EPOCHS = 1


class NCF(BaseModel):
    def __init__(self, config, logger):
        super().__init__(logger)
        device = 'cpu'
        ncf_network = NCFNetwork(config.NUM_USERS, config.NUM_MOVIES, params.EMBEDDING_SIZE).to(device)

        self.config = config
        self.ncf_network = ncf_network
        self.device = device

    def fit(self, train_data, train_predictions, **kwargs):
        train_users_torch = torch.tensor(train_data['user_id'].values, device=self.device)
        train_movies_torch = torch.tensor(train_data['movie_id'].values, device=self.device)
        train_predictions_torch = torch.tensor(train_predictions['rating'].values, device=self.device)

        train_dataloader = DataLoader(
            TensorDataset(train_users_torch, train_movies_torch, train_predictions_torch),
            batch_size=params.BATCH_SIZE)

        if self.config.TYPE == 'VAL':
            test_data = kwargs['val_data']
            test_users_torch = torch.tensor(test_data['user_id'].values, device=self.device)
            test_movies_torch = torch.tensor(test_data['movie_id'].values, device=self.device)

            test_dataloader = DataLoader(
                TensorDataset(test_users_torch, test_movies_torch),
                batch_size=params.BATCH_SIZE)

        optimizer = optim.Adam(self.ncf_network.parameters(),
                               lr=params.LEARNING_RATE)

        step = 0
        for epoch in range(params.NUM_EPOCHS):
            for users_batch, movies_batch, target_predictions_batch in train_dataloader:
                optimizer.zero_grad()
                predictions_batch = self.ncf_network(users_batch, movies_batch)
                loss = mse_loss(predictions_batch, target_predictions_batch)
                loss.backward()
                optimizer.step()
                step += 1

            if epoch % self.config.TEST_EVERY == 0 and self.config.TYPE == 'VAL':
                with torch.no_grad():
                    all_predictions = []
                    for users_batch, movies_batch in test_dataloader:
                        predictions_batch = self.ncf_network(users_batch, movies_batch)
                        all_predictions.append(predictions_batch)

                all_predictions = torch.cat(all_predictions)

                reconstuction_rmse = get_score(all_predictions.cpu().numpy(), kwargs['val_predictions'])
                self.logger.info('At epoch {:3d} loss is {:.4f}'.format(epoch, reconstuction_rmse))
                self.validation_rmse.append(reconstuction_rmse)

    def predict(self, test_data):
        test_users = test_data['user_id'].values
        test_movies = test_data['movie_id'].values
        test_users_torch = torch.tensor(test_users, device=self.device)
        test_movies_torch = torch.tensor(test_movies, device=self.device)

        test_predictions = self.ncf_network(test_users_torch, test_movies_torch).detach().numpy()
        reconstructed_matrix, _ = self.create_matrices(test_movies, test_users,
                                                       test_predictions,
                                                       default_replace='zero')

        predictions, self.index = self._extract_prediction_from_full_matrix(reconstructed_matrix, test_users,
                                                                            test_movies)
        predictions = self.postprocessing(predictions)
        return predictions

    def create_submission(self, X, suffix='', postprocessing='clipping'):
        predictions = self.postprocessing(self.predict(X), postprocessing)
        self.save_submission(self.index, predictions, suffix=suffix)
        return predictions


class NCFNetwork(nn.Module):
    def __init__(self, number_of_users, number_of_movies, embedding_size):
        super().__init__()
        self.embedding_layer_users = nn.Embedding(number_of_users, embedding_size)
        self.embedding_layer_movies = nn.Embedding(number_of_movies, embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=2 * embedding_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
            nn.ReLU()
        )

    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


def get_model(config, logger):
    return NCF(config, logger)
