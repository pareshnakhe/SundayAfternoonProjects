import random
import numpy as np
import gzip
import pickle
import jax.numpy as jnp
from jax import grad, jit, vmap, random

from functools import partial
from sklearn.utils import shuffle


"""mnist_loader"""


@jit
def vectorize(arr):
    def jnp_vectorized(j):
        return jnp.eye(10)[:, j.astype(int)].reshape(-1, 1)

    return jnp.squeeze(vmap(jnp_vectorized)(arr))


def load_data():
    f = gzip.open("./data/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    data_dict = dict()
    for data_name, dataset in zip(
        ["train", "validation", "test"], [training_data, validation_data, test_data]
    ):
        features = dataset[0]
        labels = vectorize(dataset[1])
        data_dict[data_name] = (features, labels)
    return data_dict


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""

    def scalar_sigmoid(x):
        if x > 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return np.exp(x) / (1.0 + np.exp(x))

    vfunc = np.vectorize(scalar_sigmoid)
    return vfunc(z)


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    z = np.clip(z, -100000, 100000)

    def scalar_relu(x):
        return x if x > 0 else 0

    vfunc = np.vectorize(scalar_relu)
    return vfunc(z)


def relu_prime(z):
    return (z > 0).astype(int)


def jnp_sigmoid(z):
    z = jnp.clip(z, -60, 60)
    a = 1.0 / (1.0 + jnp.exp(-z))
    b = jnp.exp(z) / (1.0 + jnp.exp(z))
    # return 1.0/(1.0+jnp.exp(-z))
    return jnp.minimum(a, b)


@jit
def vmap_batch_sigmoid(arr):
    return vmap(jnp_sigmoid)(arr)


@jit
def vmap_batch_gradient_sigmoid(arr):
    if arr.ndim != 1:
        arr_raveled = arr.ravel()
    derivative_fn = grad(jnp_sigmoid)

    return vmap(derivative_fn)(arr_raveled).reshape(arr.shape)


class SimpleNN(object):
    def __init__(self, layer_sizes):
        self.sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        self.biases = [np.random.randn(n_neurons, 1) for n_neurons in layer_sizes[1:]]
        self.weights = [
            0.8 * np.random.randn(n_neurons_1, n_neurons_2)
            for n_neurons_1, n_neurons_2 in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def feedforward(self, a):
        """
        Given input a (matrix correct dimensions).
        Returns y_hat
        """
        for weight, bias in zip(self.weights, self.biases):
            a = vmap_batch_sigmoid((a @ weight + bias.T))
            # a = sigmoid(a @ weight + bias.T)
        return a

    def compute_loss(self, input, labels):
        y_hat = self.feedforward(input)
        return np.sum(np.linalg.norm(y_hat - labels, axis=0))

    def SGD(
        self,
        train_features,
        train_labels,
        mini_batch_len=100,
        n_epochs=1,
        test_data=None,
    ):
        for epoch in range(n_epochs):
            # shuffle training set and split into mini batches
            train_features, train_labels = shuffle(train_features, train_labels)
            mini_batches = [
                (
                    train_features[i : i + mini_batch_len],
                    train_labels[i : i + mini_batch_len],
                )
                for i in range(round(len(train_features) / mini_batch_len))
            ]

            for i, mini_batch in enumerate(mini_batches):
                # run forward pass with data points in this mini-batch and update weights using backprop
                self.update_mini_batch(mini_batch)

            train_loss = self.compute_loss(train_features, train_labels)
            if test_data:
                test_loss = self.compute_loss(test_data)
                print(
                    f"Epoch {epoch}: Training Loss: {train_loss} Test Loss: {test_loss}"
                )
            else:
                print(f"Epoch {epoch}: Training Loss: {train_loss}")

    def update_mini_batch(self, mini_batch, eta=2.0):
        features = mini_batch[0]
        labels = mini_batch[1]

        nabla_b_batch, nabla_w_batch = self.backprop(features, labels)

        # update weight and bias
        self.weights = [
            wt + eta * nw / len(features) * wt
            for wt, nw in zip(self.weights, nabla_w_batch)
        ]
        self.biases = [
            bias + eta * nb.mean(axis=0).reshape(-1, 1) * bias
            for bias, nb in zip(self.biases, nabla_b_batch)
        ]

    def backprop(self, x, y):
        """
        X and y are matrices here
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = activation @ w + b.T
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.logloss_derivative(
            activations[-1], y
        ) * vmap_batch_gradient_sigmoid(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = activations[-2].T @ delta

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = vmap_batch_gradient_sigmoid(z)
            delta = (delta @ self.weights[-l + 1].T) * sp

            nabla_b[-l] = delta
            nabla_w[-l] = activations[-l - 1].T @ delta

        return (nabla_b, nabla_w)

    def squaredloss_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y

    def logloss_derivative(self, output_activations, y):
        denominator = np.clip(output_activations * (1 - output_activations), 0.01, 0.99)
        return (output_activations - y) / denominator


if __name__ == "__main__":
    data_dict = load_data()
    train_data, train_labels = data_dict["train"]

    # Change layers here
    net = SimpleNN([784, 5, 10])

    net.SGD(train_data, train_labels, mini_batch_len=1000, n_epochs=100)
