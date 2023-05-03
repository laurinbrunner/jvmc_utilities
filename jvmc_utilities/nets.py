import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jVMC.util.symmetries import LatticeSymmetry


class POVMCNN(nn.Module):
    """
    Autoregressive implementation of a Convolutional Neural Network.

    This implementation is inspired by 'WaveNet: A Generative Model for Raw Audio' by van den Oord et. al.
    (arXiv:1609.03499).
    """

    L: int = 4
    kernel_size: int = (2,)
    kernel_dilation: int = 1
    features: int = (8,)
    inputDim: int = 4
    actFun: callable = nn.elu
    depth: int = 2
    orbit: LatticeSymmetry = None
    logProbFactor: float = 1  # 1 for POVMs and 0.5 for pure wave functions

    def __call__(self, x):
        def evaluate(x):
            x_oh = jax.nn.one_hot(x, self.inputDim)
            return jnp.sum(jax.nn.log_softmax(self.cnn_cell(x_oh)) * x_oh)

        if self.orbit is None:
            # No symmetry case
            return evaluate(x)
        else:
            # Symmetry case
            x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)

            res = jnp.mean(jnp.exp(jax.vmap(evaluate)(x) / self.logProbFactor), axis=0)

            return self.logProbFactor * jnp.log(res)

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

        pad = 2 if self.depth == 0 else 1
        x = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))

        x = nn.Conv(features=4, kernel_size=self.kernel_size, kernel_dilation=1, padding='VALID')(x)

        x = self.actFun(x)

        return x[0]

    def sym_cell(self, x):
        if self.orbit is None:
            return self.cnn_cell(x)
        else:
            inShape = x.shape
            x = jax.vmap(lambda o, s: jnp.dot(o, s.ravel()).reshape(inShape), in_axes=(0, None))(self.orbit.orbit, x)

            def evaluate(x):
                return self.cnn_cell(x)

            res = jnp.mean(jax.vmap(evaluate)(x), axis=0)

            return res

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

        keys = jax.random.split(key, batchSize+1)
        configs = jax.vmap(generate_sample)(keys[:-1])

        orbitIdx = jax.random.choice(keys[-1], self.orbit.orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s), in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        return configs


class POVMCNNGated(nn.Module):
    """
    Autoregressive implementation of a Convolutional Neural Network with a gated activation function.

    This implementation is inspired by 'WaveNet: A Generative Model for Raw Audio' by van den Oord et. al.
    (arXiv:1609.03499).
    """

    L: int = 4
    kernel_size: int = (2,)
    kernel_dilation: int = 1
    features: int = (8,)
    inputDim: int = 4
    depth: int = 2
    actFun: callable = nn.elu
    orbit: LatticeSymmetry = None
    logProbFactor: float = 1  # 1 for POVMs and 0.5 for pure wave functions

    def __call__(self, x):
        def evaluate(x):
            x_oh = jax.nn.one_hot(x, self.inputDim)
            return jnp.sum(jax.nn.log_softmax(self.cnn_cell(x_oh)) * x_oh)

        if self.orbit is None:
            # No symmetry case
            return evaluate(x)
        else:
            # Symmetry case
            x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)

            res = jnp.mean(jnp.exp(jax.vmap(evaluate)(x) / self.logProbFactor), axis=0)

            return self.logProbFactor * jnp.log(res)

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

        pad = 2 if self.depth == 0 else 1
        x = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))

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

        keys = jax.random.split(key, batchSize+1)
        configs = jax.vmap(generate_sample)(keys[:-1])

        orbitIdx = jax.random.choice(keys[-1], self.orbit.orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s), in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        return configs
