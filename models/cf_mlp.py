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
        for layer in hidden_layers:
            layer.set_previous(preceding_layer)
            preceding_layer = layer

        pred_layer = tf.keras.layers.Dense(
            1, name='prediction')(preceding_layer)

        model = tf.keras.Model(
            inputs=[user_input, item_input], outputs=pred_layer)

        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='mean_squared_error',
                      metrics=['mae'])

        return model

    def train(self, x, y,
              num_factors=160, optimizer=tf.train.AdamOptimizer(),
              epochs=20, batch_size=32,
              hidden_layers=(tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(32, activation='relu'),
                             tf.keras.layers.Dropout(0.5)),
              validation_data=None, reg_p_u=0.001, reg_q_i=0.001):
        """
        Trains model on a given dataset.
        """
        if x.shape[0] != y.shape[0] or x.shape[1] != 2:
            raise ValueError('The shape of x should be (samples, 2) and '
                             'the shape of y should be (samples, 1).')

        # set parameters and hyperparameters
        self.num_factors = num_factors
        if validation_data is not None:
            x_valid, _ = validation_data
            self.num_users = max(np.max(x[:, 0]), np.max(x_valid[:, 0])) + 1
            self.num_items = max(np.max(x[:, 1]), np.max(x_valid[:, 1])) + 1
        else:
            self.num_users = np.max(x[:, 0]) + 1
            self.num_items = np.max(x[:, 1]) + 1
        self.reg_p_u = reg_p_u
        self.reg_q_i = reg_q_i

        model = self.build_model(hidden_layers)

        model.fit([x[:, 0], x[:, 1]], y, epochs=epochs, batch_size=batch_size,
                  validation_data=validation_data)
