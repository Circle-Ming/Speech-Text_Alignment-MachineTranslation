# -*- coding: utf-8 -*-

import tensorflow as tf  # 0.12
from optimizer import MaxPropOptimizer
import config

aconv1d_index = 0
# conv1d_layer
conv1d_index = 0


class VoiceModel:
    def __init__(self, batch_size, words_size, mode):
        # self._X = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, 20])
        self._X = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, config.n_mfcc])
        self._sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(self._X, reduction_indices=2), 0.),
                                                   tf.int32),
                                           reduction_indices=1)
        self._Y = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
        self._words_size = words_size
        self._mode = mode

    def train(self, sess, wav_batch, label_batch):
        # tf.logging.info(wav_batch)
        # tf.logging.info(label_batch)
        # to_return = [self._train_op, self._summaries, self._loss, self._global_step]
        #
        # to_return = [self._train_op, self._summaries, self._global_step]
        # [_, summary, step] = sess.run(to_return, feed_dict={self._X: wav_batch, self._Y: label_batch})
        # return summary, step
        to_return = [self._train_op, self._summaries, self._loss, self._global_step]
        [_, summary, loss, step] = sess.run(to_return, feed_dict={self._X: wav_batch, self._Y: label_batch})
        tf.logging.info("loss:{}\n".format(loss))
        return summary, step

    def eval(self, sess, wav_batch, label_batch):
        to_return = [self._summaries, self._loss, self._global_step]

        return sess.run(to_return,
                        feed_dict={self._X: wav_batch,
                                   self._Y: label_batch})

    def infer(self, sess, wav_batch):

        decoded = tf.transpose(self._logit, perm=[1, 0, 2])
        # decoded = self._logit
        (decoded_1, log_probabilities_1) = tf.nn.ctc_beam_search_decoder(decoded, self._sequence_len, merge_repeated=False)
        tf.logging.info(len(decoded_1))
        tf.logging.info(tf.shape(decoded_1[0]))
        tf.logging.info(log_probabilities_1)
        # predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].shape, decoded[0].values) + 1
        predict = tf.sparse_to_dense(tf.cast(decoded_1[0].indices, tf.int32), tf.shape(decoded_1[0]), decoded_1[0].values) + 1

        (decoded_2, log_probabilities_2) = tf.nn.ctc_greedy_decoder(decoded, self._sequence_len, merge_repeated=True)
        predict_2 = tf.sparse_to_dense(tf.cast(decoded_2[0].indices, tf.int32), tf.shape(decoded_2[0]), decoded_2[0].values) + 1

        to_return = [decoded_1, predict, decoded_2, predict_2, self._sequence_len, log_probabilities_1, log_probabilities_2]
        tf.logging.info(self._logit)
        return sess.run(to_return, feed_dict={self._X: wav_batch})

    def conv1d_layer(self, input_tensor, size, dim, activation, scale, bias):
        global conv1d_index
        with tf.variable_scope('conv1d_' + str(conv1d_index)):
            W = tf.get_variable('W', (size, input_tensor.get_shape().as_list()[-1], dim), dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
            if bias:
                b = tf.get_variable('b', [dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
            out = tf.nn.conv1d(input_tensor, W, stride=1, padding='SAME') + (b if bias else 0)
            if not bias:
                beta = tf.get_variable('beta', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
                gamma = tf.get_variable('gamma', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))
                mean_running = tf.get_variable('mean', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
                variance_running = tf.get_variable('variance', dim, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(1))
                # tf.logging.info("=============================================================")
                # tf.logging.info(len(out.get_shape()) - 1)
                # axes_arr = [x for x in range(len(out.get_shape()) - 1)]
                axes_arr = list(range(len(out.get_shape()) - 1))
                # tf.logging.info(axes_arr)

                mean, variance = tf.nn.moments(out, axes=axes_arr)

                def update_running_stat():
                    decay = 0.99
                    update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)),
                                 variance_running.assign(variance_running * decay + variance * (1 - decay))]
                    with tf.control_dependencies(update_op):
                        return tf.identity(mean), tf.identity(variance)
                    m, v = tf.cond(tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                                   update_running_stat, lambda: (mean_running, variance_running))
                    out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)
            if activation == 'tanh':
                out = tf.nn.tanh(out)
            if activation == 'sigmoid':
                out = tf.nn.sigmoid(out)

            conv1d_index += 1
            return out

    def aconv1d_layer(self, input_tensor, size, rate, activation, scale, bias):
        global aconv1d_index
        with tf.variable_scope('aconv1d_' + str(aconv1d_index)):
            shape = input_tensor.get_shape().as_list()
            W = tf.get_variable('W', (1, size, shape[-1], shape[-1]), dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
            if bias:
                b = tf.get_variable('b', [shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0))
            out = tf.nn.atrous_conv2d(tf.expand_dims(input_tensor, dim=1), W, rate=rate, padding='SAME')
            out = tf.squeeze(out, [1])
            if not bias:
                beta = tf.get_variable('beta', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
                gamma = tf.get_variable('gamma', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
                mean_running = tf.get_variable('mean', shape[-1], dtype=tf.float32,
                                               initializer=tf.constant_initializer(0))
                variance_running = tf.get_variable('variance', shape[-1], dtype=tf.float32,
                                                   initializer=tf.constant_initializer(1))

                # axes_arr = [x for x in range(len(out.get_shape()) - 1)]
                axes_arr = list(range(len(out.get_shape()) - 1))
                # mean, variance = tf.nn.moments(out, axes=range(len(out.get_shape()) - 1))
                mean, variance = tf.nn.moments(out, axes=axes_arr)

                def update_running_stat():
                    decay = 0.99
                    update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)),
                                 variance_running.assign(variance_running * decay + variance * (1 - decay))]
                    with tf.control_dependencies(update_op):
                        return tf.identity(mean), tf.identity(variance)
                    m, v = tf.cond(tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                                   update_running_stat, lambda: (mean_running, variance_running))
                    out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)
            if activation == 'tanh':
                out = tf.nn.tanh(out)
            if activation == 'sigmoid':
                out = tf.nn.sigmoid(out)

            aconv1d_index += 1
            return out

    # 定义神经网络
    def speech_to_text_network(self, n_dim=128, n_blocks=3):
        out = self.conv1d_layer(input_tensor=self._X, size=1, dim=n_dim, activation='tanh', scale=0.14, bias=False)

        # skip connections
        def residual_block(input_sensor, size, rate):
            conv_filter = self.aconv1d_layer(input_sensor, size=size, rate=rate, activation='tanh', scale=0.03, bias=False)
            conv_gate = self.aconv1d_layer(input_sensor, size=size, rate=rate, activation='sigmoid', scale=0.03, bias=False)
            out = conv_filter * conv_gate
            out = self.conv1d_layer(out, size=1, dim=n_dim, activation='tanh', scale=0.08, bias=False)
            return out + input_sensor, out

        skip = 0
        for _ in range(n_blocks):
            for r in [1, 2, 4, 8, 16]:
                out, s = residual_block(out, size=7, rate=r)
                skip += s

        logit = self.conv1d_layer(skip, size=1, dim=skip.get_shape().as_list()[-1], activation='tanh', scale=0.08,
                                  bias=False)
        logit = self.conv1d_layer(logit, size=1, dim=self._words_size, activation=None, scale=0.04, bias=True)
        self._logit = logit
        tf.logging.info("-------------------------self._logit-----------------")
        tf.logging.info(self._logit)
        return logit

    def calculate_loss(self):
        # CTC loss
        indices = tf.where(tf.not_equal(tf.cast(self._Y, tf.float32), 0.))
        target = tf.SparseTensor(indices=indices, values=tf.gather_nd(self._Y, indices) - 1,
                                 dense_shape=tf.cast(tf.shape(self._Y), tf.int64))
        loss = tf.nn.ctc_loss(target, self._logit, self._sequence_len, time_major=False)
        tf.summary.scalar('loss', tf.reduce_sum(loss))
        tf.summary.scalar('mean_loss', tf.reduce_mean(loss))
        return loss

    def add_train_op(self):
        # optimizer
        # lr = tf.Variable(0.4, dtype=tf.float32, trainable=False)
        lr = tf.maximum(
            0.0001,
            tf.train.exponential_decay(0.005, self._global_step, 50, 0.98))
        optimizer = MaxPropOptimizer(learning_rate=lr, beta2=0.99)
        var_list = [t for t in tf.trainable_variables()]
        gradient = optimizer.compute_gradients(self._loss, var_list=var_list)
        self._train_op = optimizer.apply_gradients(gradient, global_step=self._global_step,
                                                   name="gradient_descent_for_loss")

    def build_graph(self):
        self.speech_to_text_network()
        if self._mode != 'decode':
            self._global_step = tf.Variable(0, name='global_step', trainable=False)
            self._loss = self.calculate_loss()
        if self._mode == 'train':
            # self._lr_rate = 0.98  # FIXME delete me
            self.add_train_op()
        self._summaries = tf.summary.merge_all()
        tf.logging.info('graph built...')
