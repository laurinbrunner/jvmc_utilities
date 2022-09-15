import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


class POVMCNN(nn.Module):
    """
    Autoregressive implementation of a Convolutional Neural Network.
    """

    L: int = 4
    kernel_size: int = (2,)
    kernel_dilation: int = 1
    features: int = (8,)
    inputDim: int = 4
    actFun: callable = nn.elu
    depth: int = 2

    def __call__(self, x):
        x = jax.nn.one_hot(x, self.inputDim)
        probs = jax.nn.log_softmax(self.cnn_cell(x))

        return jnp.sum(probs * x, dtype=np.float64)

    @nn.compact
    def cnn_cell(self, x):
        x = x[:-1].reshape(1, -1, self.inputDim)

        for i in range(self.depth - 1):
            pad = 2 if i == 0 else 2**i
            x = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))

            x = nn.Conv(features=self.features[i], kernel_size=self.kernel_size,
                        kernel_dilation=2**i*self.kernel_dilation,
                        padding='VALID')(x)

            x = self.actFun(x)

        x = jnp.pad(x, ((0, 0), (1, 0), (0, 0)))

        x = nn.Conv(features=4, kernel_size=self.kernel_size, kernel_dilation=1, padding='VALID')(x)

        x = self.actFun(x)

        return x[0]

    def sample(self, batchSize, key):
        def generate_sample(key):
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=np.int64)
            conf_oh = jax.nn.one_hot(conf, self.inputDim)
            for idx in range(self.L):
                logprobs = jax.nn.log_softmax(self.cnn_cell(conf_oh)[idx].transpose()).transpose()
                conf = conf.at[idx].set(jax.random.categorical(_tmpkeys[idx], logprobs))
                conf_oh = jax.nn.one_hot(conf, self.inputDim)
            return conf

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)


class POVMCNNGated(nn.Module):
    """
    Autoregressive implementation of a Convolutional Neural Network with a gated activation function.
    """

    L: int = 4
    kernel_size: int = (2,)
    kernel_dilation: int = 1
    features: int = (8,)
    inputDim: int = 4
    depth: int = 2
    actFun: callable = nn.elu

    def __call__(self, x):
        x = jax.nn.one_hot(x, self.inputDim)
        probs = jax.nn.log_softmax(self.cnn_cell(x))

        return jnp.sum(probs * x, dtype=np.float64)

    @nn.compact
    def cnn_cell(self, x):
        x = x[:-1].reshape(1, -1, self.inputDim)

        for i in range(self.depth - 1):
            pad = 2 if i == 0 else 2**i
            x = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))

            x = nn.Conv(features=2*self.features[i], kernel_size=self.kernel_size,
                        kernel_dilation=2**i*self.kernel_dilation,
                        padding='VALID')(x)

            a, g = jnp.split(x, 2, axis=-1)
            x = nn.sigmoid(g) * jnp.tanh(a)

        x = jnp.pad(x, ((0, 0), (1, 0), (0, 0)))

        x = nn.Conv(features=4, kernel_size=self.kernel_size, kernel_dilation=1, padding='VALID')(x)

        x = self.actFun(x)

        return x[0]

    def sample(self, batchSize, key):
        def generate_sample(key):
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=np.int64)
            conf_oh = jax.nn.one_hot(conf, self.inputDim)
            for idx in range(self.L):
                logprobs = jax.nn.log_softmax(self.cnn_cell(conf_oh)[idx].transpose()).transpose()
                conf = conf.at[idx].set(jax.random.categorical(_tmpkeys[idx], logprobs))
                conf_oh = jax.nn.one_hot(conf, self.inputDim)
            return conf

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)
