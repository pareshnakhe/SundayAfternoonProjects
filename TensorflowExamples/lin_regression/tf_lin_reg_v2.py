"""Tensorflow: Linear Regression Example - Version 2

In this script, we continue with the short example from version 1 and open
the black box "minimize" to understand how it really works under the hood.

Courtesy:
Aymeric Damien (https://github.com/aymericdamien)
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/linear_regression.ipynb

Other resources:
https://www.tensorflow.org/tutorials/customization/autodiff
"""
import tensorflow as tf
import numpy as np
rng = np.random

# Parameters.
learning_rate = 0.01
training_steps = 1000
display_step = 50


def generate_dataset():
    # y = 2x + \epsilon
    x_batch = np.linspace(-1, 1, 101)
    y_batch = 2 * x_batch + rng.randn(*x_batch.shape) * 0.3

    return x_batch, y_batch


X, Y = generate_dataset()
n_samples = X.shape[0]

# Weight and Bias, initialized randomly.
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")


def linear_regression(x):
    # Linear regression (Wx + b).
    return W * x + b


def mean_square(y_pred, y_true):
    # Mean square error.
    return tf.reduce_sum(tf.pow(y_pred-y_true, 2)) / (2 * n_samples)


# Stochastic Gradient Descent Optimizer.
optimizer = tf.optimizers.SGD(learning_rate)


def run_optimization():
    """
    Tensorflow "records" all operations executed inside the context 
    of a tf.GradientTape onto a "tape". Tensorflow then uses that tape 
    and the gradients associated with each recorded operation to compute 
    the gradients of a "recorded" computation using reverse mode differentiation.
    """
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])

    # apply gradients to update the value of W and b
    optimizer.apply_gradients(zip(gradients, [W, b]))


for step in range(training_steps):
    # Run the optimization to update W and b values.
    run_optimization()

    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" %
              (step, loss, W.numpy(), b.numpy()))
