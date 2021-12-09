import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, TensorDataset

from lib.models.base_model import BaseModel
from lib.utils.utils import get_score

params = edict()
params.ENCODED_DIMENSION = 250
params.LEARNING_RATE = 0.001
params.BATCH_SIZE = 64
params.NUM_EPOCHS = 1
params.HIDDEN_DIMENSION = [500]
params.SINGLE_LAYER = False


def loss_function_autoencoder(original, reconstructed, mask):
    return torch.mean(mask * (original - reconstructed) ** 2)


class Encoder(nn.Module):
    def __init__(self, input_dimension, encoded_dimension=250):
        super().__init__()

        if params.SINGLE_LAYER:
            self.model = nn.Sequential(
                nn.Linear(in_features=input_dimension, out_features=encoded_dimension),
                nn.ReLU(),
            )
        else:
            input_layer = nn.Sequential(nn.Linear(in_features=input_dimension, out_features=params.HIDDEN_DIMENSION[0]),
                                        nn.ReLU())
            len_hidden_dimensions = len(params.HIDDEN_DIMENSION)
            if len_hidden_dimensions != 0:
                hidden_layers = [nn.Sequential(nn.Linear(in_features=params.HIDDEN_DIMENSION[i],
                                                         out_features=params.HIDDEN_DIMENSION[(
                                                                 i + 1)] if i == len_hidden_dimensions else encoded_dimension),
                                               nn.ReLU()) for i in range(len_hidden_dimensions)]
                layers = [input_layer, *hidden_layers]
            self.model = nn.Sequential(*layers)

    def forward(self, data):
        return self.model(data)


class Decoder(nn.Module):
    def __init__(self, output_dimensions, encoded_dimension=250):
        super().__init__()
        if params.SINGLE_LAYER:
            self.model = nn.Sequential(
                nn.Linear(in_features=encoded_dimension, out_features=output_dimensions),
                nn.ReLU(),
            )
        else:
            input_layer = nn.Sequential(
                nn.Linear(in_features=encoded_dimension, out_features=params.HIDDEN_DIMENSION[-1]),
                nn.ReLU())
            len_hidden_dimensions = len(params.HIDDEN_DIMENSION)
            if len_hidden_dimensions != 0:
                hidden_layers = [nn.Sequential(nn.Linear(in_features=params.HIDDEN_DIMENSION[i],
                                                         out_features=params.HIDDEN_DIMENSION[(
                                                                 i - 1)] if i != 0 else output_dimensions),
                                               nn.ReLU()) for i in range(len_hidden_dimensions - 1, -1, -1)]
                layers = [input_layer, *hidden_layers]
            self.model = nn.Sequential(*layers)

    def forward(self, encodings):
        return self.model(encodings)


class AutoEncoderNetwork(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(self.encoder(data))


class AutoEncoder(BaseModel):
    def __init__(self, config, logger, encoded_dimension=params.ENCODED_DIMENSION, single_layer=params.SINGLE_LAYER,
                 hidden_dimension=params.HIDDEN_DIMENSION):

        super().__init__(logger)
        self.config = config

        params.ENCODED_DIMENSION = encoded_dimension
        params.SINGLE_LAYER = single_layer
        params.HIDDEN_DIMENSION = hidden_dimension
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder_network = AutoEncoderNetwork(
            encoder=Encoder(
                input_dimension=self.config.NUM_MOVIES,
                encoded_dimension=params.ENCODED_DIMENSION,
            ),
            decoder=Decoder(
                output_dimensions=config.NUM_MOVIES,
                encoded_dimension=params.ENCODED_DIMENSION,
            )
        ).to(device)

        self.autoencoder_network = autoencoder_network
        self.logger = logger
        self.num_users = config.NUM_USERS
        self.num_movies = config.NUM_MOVIES
        self.num_epochs = params.NUM_EPOCHS
        self.batch_size = params.BATCH_SIZE
        self.device = device
        self.loss_function = loss_function_autoencoder
        self.optimizer = optim.Adam(self.autoencoder_network.parameters(), lr=params.LEARNING_RATE)

    def fit(self, train_data, train_predictions, **kwargs):
        # Build Dataloaders
        data, mask = self.create_matrices(train_data['movie_id'].values, train_data['user_id'].values,
                                          train_predictions['rating'].values)
        self.data_torch = torch.tensor(data, device=self.device).float()
        self.mask_torch = torch.tensor(mask, device=self.device)

        dataloader = DataLoader(
            TensorDataset(self.data_torch, self.mask_torch),
            batch_size=self.batch_size)
        step = 0
        for epoch in range(self.num_epochs):
            for data_batch, mask_batch in dataloader:
                self.optimizer.zero_grad()

                reconstructed_batch = self.autoencoder_network(data_batch)

                loss = self.loss_function(data_batch, reconstructed_batch, mask_batch)

                loss.backward()

                self.optimizer.step()
                step += 1

            if epoch % self.config.TEST_EVERY == 0 and self.config.TYPE == 'VAL':
                predictions = self.predict(kwargs['val_data'])
                reconstruction_rmse = get_score(predictions, kwargs['val_predictions'])
                self.logger.info('At epoch {:3d} loss is {:.4f}'.format(epoch, reconstruction_rmse))
                self.validation_rmse.append(reconstruction_rmse)

    def predict(self, test_data):
        reconstructed_matrix = self.reconstruct_whole_matrix()
        predictions, self.index = self._extract_prediction_from_full_matrix(reconstructed_matrix,
                                                                            users=test_data['user_id'].values,
                                                                            movies=test_data['movie_id'].values)
        predictions = self.postprocessing(predictions)
        return predictions

    def create_submission(self, X, suffix='', postprocessing='clipping'):
        predictions = self.postprocessing(self.predict(X), postprocessing)
        self.save_submission(self.index, predictions, suffix=suffix)
        return predictions

    def reconstruct_whole_matrix(self):
        data_reconstructed = np.zeros((self.num_users, self.num_movies))

        with torch.no_grad():
            for i in range(0, self.num_users, self.batch_size):
                upper_bound = min(i + self.batch_size, self.num_users)
                data_reconstructed[i:upper_bound] = self.autoencoder_network(
                    self.data_torch[i:upper_bound]).detach().cpu().numpy()

        return data_reconstructed


def get_model(config, logger):
    return AutoEncoder(config, logger)
