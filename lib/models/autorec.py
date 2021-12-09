'''
Adapted from https://github.com/gtshs2/Autorec
'''
import argparse
import math
import os
import time

import numpy as np
import tensorflow as tf

from lib.models.base_model import BaseModel

tf1 = tf.compat.v1
tf1.disable_v2_behavior()


class AutoRec(BaseModel):
    def __init__(self, config, logger):

        super().__init__(logger)
        assert config.TYPE == 'VAL', "Use validation mode 'VAL'"
        self.config = config

    def run(self):
        self.prepare_model()
        init = tf1.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(self.train_epoch):
            self.train_model(epoch_itr)
            self.test_model(epoch_itr)
        self.make_records()

    def prepare_model(self):
        self.input_R = tf1.placeholder(dtype=tf1.float32, shape=[None, self.num_items], name="input_R")
        self.input_mask_R = tf1.placeholder(dtype=tf1.float32, shape=[None, self.num_items], name="input_mask_R")

        V = tf1.get_variable(name="V", initializer=tf1.truncated_normal(shape=[self.num_items, self.hidden_neuron],
                                                                        mean=0, stddev=0.03), dtype=tf1.float32)
        W = tf1.get_variable(name="W", initializer=tf1.truncated_normal(shape=[self.hidden_neuron, self.num_items],
                                                                        mean=0, stddev=0.03), dtype=tf1.float32)
        mu = tf1.get_variable(name="mu", initializer=tf1.zeros(shape=self.hidden_neuron), dtype=tf1.float32)
        b = tf1.get_variable(name="b", initializer=tf1.zeros(shape=self.num_items), dtype=tf1.float32)

        pre_Encoder = tf1.matmul(self.input_R, V) + mu
        self.Encoder = tf1.nn.sigmoid(pre_Encoder)
        pre_Decoder = tf1.matmul(self.Encoder, W) + b
        self.Decoder = tf1.identity(pre_Decoder)

        pre_rec_cost = tf1.multiply((self.input_R - self.Decoder), self.input_mask_R)
        rec_cost = tf1.square(self.l2_norm(pre_rec_cost))
        pre_reg_cost = tf1.square(self.l2_norm(W)) + tf1.square(self.l2_norm(V))
        reg_cost = self.lambda_value * 0.5 * pre_reg_cost

        self.cost = rec_cost + reg_cost

        if self.optimizer_method == "Adam":
            optimizer = tf1.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf1.train.RMSPropOptimizer(self.lr)
        else:
            raise ValueError("Optimizer Key ERROR")

        if self.grad_clip:
            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf1.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        else:
            self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)

    def train_model(self, itr):
        start_time = time.time()
        random_perm_doc_idx = np.random.permutation(self.num_users)

        batch_cost = 0
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size: (i + 1) * self.batch_size]

            _, Cost = self.sess.run(
                [self.optimizer, self.cost],
                feed_dict={self.input_R: self.train_R[batch_set_idx, :],
                           self.input_mask_R: self.train_mask_R[batch_set_idx, :]})

            batch_cost = batch_cost + Cost
        self.train_cost_list.append(batch_cost)

        if (itr) % self.display_step == 0:
            print("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(batch_cost),
                  "Elapsed time : %d sec" % (time.time() - start_time))

    def test_model(self, itr):
        start_time = time.time()
        Cost, Decoder = self.sess.run(
            [self.cost, self.Decoder],
            feed_dict={self.input_R: self.train_R,
                       self.input_mask_R: self.train_mask_R})

        self.test_cost_list.append(Cost)

        if (itr) % self.display_step == 0:
            Estimated_R = Decoder.clip(min=1, max=5)
            unseen_user_test_list = list(self.user_test_set - self.user_train_set)
            unseen_item_test_list = list(self.item_test_set - self.item_train_set)

            for user in unseen_user_test_list:
                for item in unseen_item_test_list:
                    if self.test_mask_R[user, item] == 1:  # exist in test set
                        Estimated_R[user, item] = 3

            pre_numerator = np.multiply((Estimated_R - self.test_R), self.test_mask_R)
            numerator = np.sum(np.square(pre_numerator))
            denominator = self.num_test_ratings
            RMSE = np.sqrt(numerator / float(denominator))

            self.test_rmse_list.append(RMSE)
            self.validation_rmse.append(RMSE)

            print("Testing //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(Cost),
                  " RMSE = {:.5f}".format(RMSE),
                  "Elapsed time : %d sec" % (time.time() - start_time))
            print("=" * 100)

    def fit(self, train_data, train_predictions, **kwargs):
        parser = argparse.ArgumentParser(description='I-AutoRec ')
        parser.add_argument('--hidden_neuron', type=int, default=500)
        parser.add_argument('--lambda_value', type=float, default=1)

        parser.add_argument('--train_epoch', type=int, default=125)
        parser.add_argument('--batch_size', type=int, default=100)

        parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
        parser.add_argument('--grad_clip', type=bool, default=False)
        parser.add_argument('--base_lr', type=float, default=1e-3)
        parser.add_argument('--decay_epoch_step', type=int, default=50,
                            help="decay the learning rate for each n epochs")

        parser.add_argument('--random_seed', type=int, default=1000)
        parser.add_argument('--display_step', type=int, default=5)

        self.args = parser.parse_args()
        tf1.set_random_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)

        self.configuration = tf1.ConfigProto()
        self.configuration.gpu_options.allow_growth = True

        val_data, val_predictions = kwargs['val_data'], kwargs['val_predictions']
        data = train_data.append(val_data)
        predictions = np.concatenate([train_predictions['rating'].values, val_predictions])

        self.num_users = self.config.NUM_USERS
        self.num_items = self.config.NUM_MOVIES

        self.R, self.mask_R = self.create_matrices(data['movie_id'], data['user_id'].values, predictions,
                                                   default_replace='zero')
        self.C = self.mask_R.copy()
        self.train_R, self.train_mask_R = self.create_matrices(train_data['movie_id'].values,
                                                               train_data['user_id'].values,
                                                               train_predictions['rating'].values,
                                                               default_replace='zero')
        self.test_R, self.test_mask_R = self.create_matrices(val_data['movie_id'].values, val_data['user_id'].values,
                                                             val_predictions,
                                                             default_replace='zero')
        self.num_train_ratings = train_data.shape[0]
        self.num_test_ratings = val_data.shape[0]

        self.user_train_set = set(train_data['user_id'].values)
        self.item_train_set = set(train_data['movie_id'].values)
        self.user_test_set = set(val_data['user_id'].values)
        self.item_test_set = set(val_data['movie_id'].values)

        self.hidden_neuron = self.args.hidden_neuron
        self.train_epoch = self.args.train_epoch
        self.batch_size = self.args.batch_size
        self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))

        self.base_lr = self.args.base_lr
        self.optimizer_method = self.args.optimizer_method
        self.display_step = self.args.display_step
        self.random_seed = self.args.random_seed

        self.global_step = tf1.Variable(0, trainable=False)
        self.decay_epoch_step = self.args.decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch
        self.lr = tf1.train.exponential_decay(self.base_lr, self.global_step,
                                              self.decay_step, 0.96, staircase=True)
        self.lambda_value = self.args.lambda_value

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []

        self.result_path = "output/autorec/"
        self.grad_clip = self.args.grad_clip

        self.sess = tf1.Session(config=self.configuration)
        self.run()

    def predict(self, test_data):
        Cost, Decoder = self.sess.run(
            [self.cost, self.Decoder],
            feed_dict={self.input_R: self.R,
                       self.input_mask_R: self.mask_R})

        Estimated_R = Decoder.clip(min=1, max=5)
        predictions, self.index = self._extract_prediction_from_full_matrix(Estimated_R,
                                                                            users=test_data['user_id'].values,
                                                                            movies=test_data['movie_id'].values)
        predictions = self.postprocessing(predictions)
        return predictions

    def create_submission(self, X, suffix='', postprocessing='clipping'):
        predictions = self.postprocessing(self.predict(X), postprocessing)
        self.save_submission(self.index, predictions, suffix)
        return predictions

    def make_records(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        basic_info = self.result_path + "basic_info.txt"
        train_record = self.result_path + "train_record.txt"
        test_record = self.result_path + "test_record.txt"

        with open(train_record, 'w') as f:
            f.write(str("Cost:"))
            f.write('\t')
            for itr in range(len(self.train_cost_list)):
                f.write(str(self.train_cost_list[itr]))
                f.write('\t')
            f.write('\n')

        with open(test_record, 'w') as g:
            g.write(str("Cost:"))
            g.write('\t')
            for itr in range(len(self.test_cost_list)):
                g.write(str(self.test_cost_list[itr]))
                g.write('\t')
            g.write('\n')

            g.write(str("RMSE:"))
            for itr in range(len(self.test_rmse_list)):
                g.write(str(self.test_rmse_list[itr]))
                g.write('\t')
            g.write('\n')

        with open(basic_info, 'w') as h:
            h.write(str(self.args))

    def l2_norm(self, tensor):
        return tf1.sqrt(tf1.reduce_sum(tf1.square(tensor)))


def get_model(config, logger):
    return AutoRec(config, logger)
