# -*- coding: utf-8 -*-

import tensorflow as tf


class MaxPropOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta2=0.999, use_locking=False, name="MaxProp"):
        super(MaxPropOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta2 = beta2
        self._lr_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7
        else:
            eps = 1e-8
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = grad / m_t
        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)
