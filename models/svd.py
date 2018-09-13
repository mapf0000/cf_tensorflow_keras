import numpy as np
import tensorflow as tf


class SVD:
    """
    An implementation of the SVD recommeder model,
    see http://sifter.org/simon/journal/20061211.html
    """

    def __init__(self, session):
        self.num_factors = None
        self.num_users = None
        self.num_items = None
        self.mu = None
        self.reg_p_u = None
        self.reg_b_u = None
        self.reg_q_i = None
        self.reg_b_i = None
        self.sess = session

    def build_model(self, users, items, ratings):
        """ 
        Builds model graph.
        """
        # constants
        with tf.variable_scope('constants', reuse=tf.AUTO_REUSE):
            _mu = tf.constant(self.mu, shape=[], dtype=tf.float32)

        # user variables
        with tf.variable_scope('users', reuse=tf.AUTO_REUSE):
            user_embeddings = tf.get_variable(
                name='embeddings',
                shape=[self.num_users, self.num_factors],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.reg_p_u))

            user_bias = tf.get_variable(
                name='bias',
                shape=[self.num_users, ],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.reg_b_u))

            p_u = tf.nn.embedding_lookup(
                user_embeddings,
                users,
                name='p_u')

            b_u = tf.nn.embedding_lookup(
                user_bias,
                users,
                name='b_u')

        # item variables
        with tf.variable_scope('items', reuse=tf.AUTO_REUSE):
            item_embeddings = tf.get_variable(
                name='embeddings',
                shape=[self.num_items, self.num_factors],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.reg_q_i))

            item_bias = tf.get_variable(
                name='bias',
                shape=[self.num_items, ],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.reg_b_i))

            q_i = tf.nn.embedding_lookup(
                item_embeddings,
                items,
                name='q_i')

            b_i = tf.nn.embedding_lookup(
                item_bias,
                items,
                name='b_i')

        # prediction
        with tf.variable_scope('prediction', reuse=tf.AUTO_REUSE):
            prediction = tf.reduce_sum(tf.multiply(p_u, q_i), axis=1)
            prediction = tf.add_n([b_u, b_i, prediction])
            prediction = tf.add(prediction, _mu, name='prediction')

        # loss
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss = tf.nn.l2_loss(tf.subtract(ratings, prediction), name='loss')

            training_objective = tf.add(loss,
                                        tf.add_n(tf.get_collection(
                                            tf.GraphKeys.REGULARIZATION_LOSSES)),
                                        name='objective')

        return training_objective, prediction

    def create_metrics(self, pred, ratings):
        """ 
        Creates evaluation Metrics and update OPs.
        """
        with tf.variable_scope('metrics', reuse=tf.AUTO_REUSE):
            mae, mae_update_op = tf.metrics.mean_absolute_error(
                ratings, pred, name="mae")

            rmse, rmse_update_op = tf.metrics.root_mean_squared_error(
                tf.cast(ratings, tf.float32), tf.cast(pred, tf.float32), name="rmse")

        return mae, mae_update_op, rmse, rmse_update_op

    def train(self, x, y,
              num_factors=50, optimizer=tf.train.AdamOptimizer(),
              validation_data=None,
              batch_size=1024, epochs=10, early_stopping=None,
              reg_p_u=0.0001, reg_b_u=0.0001,
              reg_q_i=0.0001, reg_b_i=0.0001):
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
        self.mu = np.mean(y)
        self.reg_b_u = reg_b_u
        self.reg_b_i = reg_b_i
        self.reg_p_u = reg_p_u
        self.reg_q_i = reg_q_i

        # create datasets
        training_dataset = tf.data.Dataset.from_tensor_slices(
            (x[:, 0].astype(np.int32),
             x[:, 1].astype(np.int32),
             y.astype(np.float32))).batch(batch_size)

        if validation_data is not None:
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (x_valid[:, 0].astype(np.int32),
                 x_valid[:, 1].astype(np.int32),
                 y_valid.astype(np.float32))).batch(batch_size)

        # create dataset iterator and OPs
        dataset_iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                                           training_dataset.output_shapes)

        users, items, ratings = dataset_iterator.get_next()

        training_init_op = dataset_iterator.make_initializer(training_dataset)
        if validation_data is not None:
            validation_init_op = dataset_iterator.make_initializer(
                validation_dataset)

        # build model
        training_objective, prediction = self.build_model(
            users, items, ratings)
        optimizer = optimizer.minimize(training_objective, name='optimizer')

        if validation_data is not None:
            mae, mae_update_op, rmse, rmse_update_op = self.create_metrics(
                prediction, ratings)

        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            best_loss = None

            print('Training...')
            for epoch in range(0, epochs):
                print('Epoch {}/{}'.format(epoch + 1, epochs))

                # training
                sess.run(training_init_op)
                training_loss = 0.0
                counter = 0

                while True:
                    try:
                        _, loss_value = sess.run(
                            [optimizer, training_objective])
                        training_loss += loss_value/batch_size
                        counter += 1
                    except tf.errors.OutOfRangeError:
                        # after last batch
                        break

                print("Training loss: {:.4f}"
                      .format(training_loss/counter))

                # validation
                if validation_data is not None:
                    sess.run(tf.local_variables_initializer())
                    sess.run(validation_init_op)
                    valid_loss = 0.0
                    counter = 0

                    while True:
                        try:
                            loss_value, _, _ = sess.run(
                                [training_objective, mae_update_op, rmse_update_op])
                            valid_loss += loss_value/batch_size
                            counter += 1
                        except tf.errors.OutOfRangeError:
                            # after last batch
                            break

                    mae_val, rmse_val = sess.run([mae, rmse])
                    print("Validation loss: {:.4f} | MAE: {:.4f} | RMSE: {:.4f}"
                          .format(valid_loss/counter, mae_val, rmse_val))

                    if early_stopping is not None:
                        if best_loss is None:
                            best_loss = valid_loss
                        else:
                            if valid_loss < best_loss:
                                best_loss = valid_loss
                                saver.save(sess, "tmp/best_model.ckpt")
                            else:
                                early_stopping -= 1
                                if early_stopping < 0:
                                    print(
                                        "Validation loss did not improve --> stopping.")
                                    saver.restore(sess, "tmp/best_model.ckpt")
                                    break
                                else:
                                    print(
                                        "Validation loss did not improve --> patience = {:0d}."
                                        .format(early_stopping))

            saver.save(sess, "logdir/model.ckpt")

    def predict(self, user, item):
        # TODO
        return
