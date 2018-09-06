import numpy as np
import tensorflow as tf


class Cf_mlp():
    """
    An implementation of the MLP Neural Collaborative Filtering Model
    in https://arxiv.org/pdf/1708.05031.pdf.
    """

    def __init__(self):
        self.num_factors = None
        self.num_users = None
        self.num_items = None
        self.reg_p_u = None
        self.reg_q_i = None

    def build_model(self, hidden_layers):
        """
        Builds the model graph.
        """
        user_input = tf.keras.Input(
            shape=(1,), dtype='int32', name='user_input')
        item_input = tf.keras.Input(
            shape=(1,), dtype='int32', name='item_input')

        user_embeddings = tf.keras.layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.num_factors,
            name='user_embeddings',
            embeddings_initializer=tf.keras.initializers.glorot_normal(),
            embeddings_regularizer=tf.keras.regularizers.l2(self.reg_p_u),
            input_length=1)

        item_embeddings = tf.keras.layers.Embedding(
            input_dim=self.num_items,
            output_dim=self.num_factors,
            name='item_embeddings',
            embeddings_initializer=tf.keras.initializers.glorot_normal(),
            embeddings_regularizer=tf.keras.regularizers.l2(self.reg_q_i),
            input_length=1)

        concat_input = tf.keras.layers.concatenate(
            [tf.keras.layers.Flatten()(user_embeddings(user_input)),
             tf.keras.layers.Flatten()(item_embeddings(item_input))])

        preceding_layer = concat_input
        for layer_description in hidden_layers:
            l_name, l_param = layer_description
            if l_name is "dense":
                preceding_layer = tf.keras.layers.Dense(l_param)(preceding_layer)
            if l_name is "dropout":
                preceding_layer = tf.keras.layers.Dropout(l_param)(preceding_layer)

        pred_layer = tf.keras.layers.Dense(
            1, name='prediction')(preceding_layer)

        model = tf.keras.Model(
            inputs=[user_input, item_input], outputs=pred_layer)

        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='mean_squared_error',
                      metrics=['mae'])

        return model

    def train(self, x, y,
              num_factors=200, optimizer=tf.train.AdamOptimizer(),
              epochs=20, batch_size=32,
              hidden_layers=[("dense", 128),
                             ("dropout", 0.5),
                             ("dense", 64),
                             ("dropout", 0.5),
                             ("dense", 32),
                             ("dropout", 0.5)],
              validation_data=None, reg_p_u=0.001, reg_q_i=0.001,
              early_stopping=None):
        """
        Trains model on a given dataset.
        """
        if x.shape[0] != y.shape[0] or x.shape[1] != 2:
            raise ValueError('The shape of x should be (samples, 2) and '
                             'the shape of y should be (samples, 1).')

        # set parameters and hyperparameters
        self.num_factors = num_factors
        if validation_data is not None:
            x_valid, y_valid = validation_data
            self.num_users = max(np.max(x[:, 0]), np.max(x_valid[:, 0])) + 1
            self.num_items = max(np.max(x[:, 1]), np.max(x_valid[:, 1])) + 1
        else:
            self.num_users = np.max(x[:, 0]) + 1
            self.num_items = np.max(x[:, 1]) + 1
        self.reg_p_u = reg_p_u
        self.reg_q_i = reg_q_i

        model = self.build_model(hidden_layers)

        if early_stopping is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(patience=early_stopping, verbose=1)
        model.fit([x[:, 0], x[:, 1]], y, epochs=epochs, batch_size=batch_size,
                  validation_data=([x_valid[:, 0], x_valid[:, 1]], y_valid), callbacks=[early_stopping])