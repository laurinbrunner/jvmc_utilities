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

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)


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

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)


class AFFN(nn.Module):
    """
    Autoregressive implementation of a feed forward neural network.

    This implementation is inspired by 'Solving Statistical Mechanics Using Variational Autoregressive Networks'
    DOI: 10.1103/PhysRevLett.122.080602
    """

    L: int = 4
    hiddenSize: int = 8
    inputDim: int = 4
    depth: int = 2
    actFun: callable = nn.elu

    def __call__(self, x):
        x = nn.one_hot(x, self.inputDim)
        probs = nn.log_softmax(self.fnn_cell(x))

        return jnp.sum(probs * x, dtype=np.float64)

    @nn.compact
    def fnn_cell(self, x):
        mask = jnp.triu(jnp.ones((x.shape[0], self.hiddenSize)))

        W = nn.Dense(features=self.hiddenSize, use_bias=True)(jnp.ones_like(x))
        W = jnp.multiply(mask, W)

        x = jnp.dot(W, x)

        return self.actFun(x)

    def sample(self, batchSize, key):
        def generate_sample(key):
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=np.int64)
            conf_oh = jax.nn.one_hot(conf, self.inputDim)
            for idx in range(self.L):
                logprobs = jax.nn.log_softmax(self.fnn_cell(conf_oh)[idx].transpose()).transpose()
                conf = conf.at[idx].set(jax.random.categorical(_tmpkeys[idx], logprobs))
                conf_oh = jax.nn.one_hot(conf, self.inputDim)
            return conf

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)


class DeepNADE(nn.Module):
    """
    Implementation of a deep Neural Autoregressive Distribution Estimation model.

    This implementation is inspired by 'Neural Autoregressive Distribution Estimation' (arXiv:1605.02226).
    """

    L: int = 4
    hiddenSize: int = 8
    inputDim: int = 4
    depth: int = 2
    actFun: callable = nn.elu
    orbit: LatticeSymmetry = None
    logProbFactor: float = 1.  # 1 for POVMs and 0.5 for pure wave functions

    def setup(self) -> None:
        self.deep_layers = [[nn.Dense(features=self.hiddenSize, use_bias=True if (i == 0 and _ == 0) else False)
                             for i in range(self.depth)] for _ in range(self.L)]
        self.last_layer = [nn.Dense(features=4, use_bias=True) for _ in range(self.L)]

    def __call__(self, x):
        def evaluate(x):
            x_oh = nn.one_hot(x, self.inputDim)
            return jnp.sum(nn.log_softmax(self.nade_cell(x_oh)) * x_oh)

        if self.orbit is not None:
            # Symmetry case
            x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)
            res = jnp.mean(jnp.exp(jax.vmap(evaluate)(x) / self.logProbFactor), axis=0)

            return self.logProbFactor * jnp.log(res)
        else:
            # No symmetry case
            return evaluate(x)

    @nn.compact
    def nade_cell(self, x):
        p = jnp.zeros_like(x, dtype=np.float32)
        x = x[:-1].reshape(1, -1, self.inputDim)

        x = jnp.pad(x, ((0, 0), (1, 0), (0, 0)))
        a = jnp.zeros((self.depth, self.hiddenSize))
        for idx in range(self.L):
            da = x[0, idx]
            for i in range(self.depth):
                if i == 0:
                    da = self.deep_layers[idx][i](da)
                else:
                    da = self.deep_layers[idx][i](self.actFun(da))
                da += a[i]
                a = a.at[i].set(da)

            k = self.last_layer[idx](da)
            k = self.actFun(k)
            p = p.at[idx].set(k)

        return p

    def sample(self, batchSize, key):
        def generate_sample(key):
            _tmpkeys = jax.random.split(key, self.L)

            conf = jnp.zeros(self.L, dtype=int)
            a = jnp.zeros((self.depth, self.hiddenSize))
            previous_site = jnp.zeros(self.inputDim)
            for idx in range(self.L):
                da = previous_site
                for i in range(self.depth):
                    if i == 0:
                        da = self.deep_layers[idx][i](da)
                    else:
                        da = self.deep_layers[idx][i](self.actFun(da))
                    da += a[i]
                    a = a.at[i].set(da)

                k = self.actFun(self.last_layer[idx](da))

                logprobs = nn.log_softmax(k.transpose()).transpose()
                new_site = jax.random.categorical(_tmpkeys[idx], logprobs)
                conf = conf.at[idx].set(new_site)
                previous_site = nn.one_hot(new_site, self.inputDim)

            return conf

        keys = jax.random.split(key, batchSize + 1)
        configs = jax.vmap(generate_sample)(keys[:-1])

        if self.orbit is not None:
            orbitIdx = jax.random.choice(keys[-1], self.orbit.orbit.shape[0], shape=(batchSize,))
            configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s),
                               in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        return configs