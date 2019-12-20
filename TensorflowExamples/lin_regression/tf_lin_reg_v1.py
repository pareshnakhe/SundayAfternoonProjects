"""Tensorflow: Linear Regression Example - Version 1

In this script I have implemented the most basic version
of linear regression example.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def generate_dataset():
    # y = 2x + \epsilon
    x_batch = np.linspace(-1, 1, 101)
    y_batch = 2 * x_batch + np.random.randn(*x_batch.shape) * 0.3

    return x_batch, y_batch


def lin_reg(data):
    x = data[0]
    y = data[1]
    w = tf.Variable(np.random.normal(), dtype=tf.float64, name='W')

    def compute_loss():
        y_pred = tf.multiply(w, x)
        return tf.reduce_sum(tf.square(y_pred - y))

    def _compute_loss(w):
        y_pred = tf.multiply(w, x)
        return tf.reduce_sum(tf.square(y_pred - y))

    for _ in range(10):
        print("loss: {}, w: {}".format(_compute_loss(w).numpy(), w.numpy()))
        tf.optimizers.SGD().minimize(compute_loss, var_list=[w])

    return w


x, y = generate_dataset()
out_a = lin_reg([x, y])
