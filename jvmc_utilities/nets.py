import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jVMC.util.symmetries import LatticeSymmetry
from jax._src.prng import PRNGKeyArray
from typing import Union, Tuple


class POVMCNN(nn.Module):
    """
    Autoregressive implementation of a Convolutional Neural Network.

    This implementation is inspired by 'WaveNet: A Generative Model for Raw Audio' by van den Oord et. al.
    (arXiv:1609.03499).

    :param L: system size
    :param kernel_size: size of kernel for convolution
    :param features: number of hidden units
    :param inputDim: dimension of the input (4 for POVMs and 2 for spin 1/2)
    :param depth: number of convolutional layers
    :param actFun: activation function to be used at the end of every layer
    :param orbit: LatticeSymmetry object that encodes all symmetry transformations applicable to the system
    :param logProbFactor: exponent of the probability (1 for POVMs and 0.5 for pure wave functions)
    :param param_dtype: data type for network parameters
    """

    L: int = 4
    kernel_size: Tuple[int] = (2,)
    features: Union[Tuple[int], int] = 8
    inputDim: int = 4
    actFun: callable = nn.elu
    depth: int = 2
    orbit: LatticeSymmetry = None
    logProbFactor: float = 1.  # 1 for POVMs and 0.5 for pure wave functions
    param_dtype: type = jnp.float32

    def setup(self) -> None:
        features = self.features
        if type(self.features) is int:
            features = tuple([self.features] * self.depth)
        self.conv_cells = [CNNCell(features=features[i] if i != self.depth - 1 else self.inputDim,
                                   kernel_size=self.kernel_size, actFun=self.actFun, param_dtype=self.param_dtype,
                                   kernel_dilation=self.kernel_size[0]**i)
                           for i in range(self.depth)]

        self.paddings = tuple([self.kernel_size[0]] +
                              [self.kernel_size[0]**i * (self.kernel_size[0] - 1) for i in range(1, self.depth)])

        self.cache_sizes = tuple([(self.kernel_size[0], self.inputDim)] +
                                 [(self.paddings[i] + 1, features[i - 1]) for i in range(1, self.depth)])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        def evaluate(x: jnp.ndarray):
            x_oh = jax.nn.one_hot(x, self.inputDim)
            return jnp.sum(jax.nn.log_softmax(self.cnn_cell(x_oh)) * x_oh)

        if self.orbit is not None:
            # Symmetry case
            x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)

            res = jnp.mean(jnp.exp(jax.vmap(evaluate)(x) / self.logProbFactor), axis=0)

            return self.logProbFactor * jnp.log(res)
        else:
            # No symmetry case
            return evaluate(x)

    def cnn_cell(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x[:-1].reshape(1, -1, self.inputDim)

        for i in range(self.depth):
            x = jnp.pad(x, ((0, 0), (self.paddings[i], 0), (0, 0)))
            x = self.conv_cells[i](x)

        return x[0]

    def sample(self, batchSize: int, key: PRNGKeyArray) -> jnp.ndarray:
        """
        This implementation is inspired by 'Fast Generation for Convolutional Autoregressive Models' (arXiv:1704.06001).
        """
        def generate_sample(key: PRNGKeyArray) -> jnp.ndarray:
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=np.int64)

            # This list caches the input to the i-th cnn layer
            cache = [jnp.zeros(rf) for rf in self.cache_sizes]

            for idx in range(self.L):
                for i in range(self.depth):
                    x = jnp.copy(cache[i])
                    x = self.conv_cells[i](x)

                    if i != self.depth - 1:
                        cache[i+1] = jnp.roll(cache[i+1], -1, axis=0)
                        cache[i+1] = cache[i+1].at[-1].set(x[0])

                new_value = jax.random.categorical(_tmpkeys[idx], nn.log_softmax(x[0].transpose()).transpose())
                conf = conf.at[idx].set(new_value)
                cache[0] = jnp.roll(cache[0], -1, axis=0)
                cache[0] = cache[0].at[-1].set(nn.one_hot(new_value, self.inputDim))

            return conf

        keys = jax.random.split(key, batchSize+1)
        configs = jax.vmap(generate_sample)(keys[:-1])

        if self.orbit is not None:
            orbitIdx = jax.random.choice(keys[-1], self.orbit.orbit.shape[0], shape=(batchSize,))
            configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s),
                               in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        return configs


class POVMCNNGated(POVMCNN):
    """
    Autoregressive implementation of a Convolutional Neural Network with a gated activation function.

    This implementation is inspired by 'WaveNet: A Generative Model for Raw Audio' by van den Oord et. al.
    (arXiv:1609.03499).

    :param L: system size
    :param kernel_size: size of kernel for convolution
    :param features: number of hidden units
    :param inputDim: dimension of the input (4 for POVMs and 2 for spin 1/2)
    :param depth: number of NADE layers
    :param actFun: activation function to be used at the end of every layer
    :param orbit: LatticeSymmetry object that encodes all symmetry transformations applicable to the system
    :param logProbFactor: exponent of the probability (1 for POVMs and 0.5 for pure wave functions)
    :param param_dtype: data type for network parameters
    """

    def setup(self) -> None:
        features = self.features
        if type(self.features) is int:
            features = tuple([self.features] * self.depth)
        # noinspection PyTypeChecker
        self.conv_cells = [GatedCNNCell(features=2 * features[i],
                                        kernel_size=self.kernel_size, param_dtype=self.param_dtype,
                                        kernel_dilation=self.kernel_size[0] ** i)
                           for i in range(self.depth - 1)] + \
                          [CNNCell(features=self.inputDim, kernel_size=self.kernel_size, param_dtype=self.param_dtype,
                                   kernel_dilation=self.kernel_size[0] ** (self.depth - 1))]

        self.paddings = tuple([self.kernel_size[0]] +
                              [self.kernel_size[0] ** i * (self.kernel_size[0] - 1) for i in range(1, self.depth)])

        self.cache_sizes = tuple([(self.kernel_size[0], self.inputDim)] +
                                 [(self.paddings[i] + 1, features[i - 1]) for i in range(1, self.depth)])


class POVMCNNResidual(POVMCNN):

    def setup(self) -> None:
        super().setup()
        features = self.features
        if type(self.features) is int:
            features = tuple([self.features] * self.depth)
        self.residual_convs = [nn.Conv(features=features[i] if i != self.depth - 1 else self.inputDim,
                                       kernel_size=(1,), param_dtype=self.param_dtype)
                               for i in range(self.depth)]

    def cnn_cell(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(1, -1, self.inputDim)
        x_omitted = x[:, :-1]

        for i in range(self.depth):
            x_padded = jnp.pad(x if i != 0 else x_omitted, ((0, 0), (self.paddings[i], 0), (0, 0)))
            x = self.actFun(self.conv_cells[i](x_padded) + self.residual_convs[i](x))

        return x[0]

    def sample(self, batchSize: int, key) -> jnp.ndarray:
        """
        This implementation is inspired by 'Fast Generation for Convolutional Autoregressive Models' (arXiv:1704.06001).
        """
        def generate_sample(key) -> jnp.ndarray:
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=np.int64)

            # This list caches the input to the i-th cnn layer
            cache = [jnp.zeros(rf) for rf in self.cache_sizes]

            for idx in range(self.L):
                for i in range(self.depth):
                    x = jnp.copy(cache[i])
                    x = self.actFun(self.conv_cells[i](x) + self.residual_convs[i](x))

                    if i != self.depth - 1:
                        cache[i+1] = jnp.roll(cache[i+1], -1, axis=0)
                        cache[i+1] = cache[i+1].at[-1].set(x[0])

                new_value = jax.random.categorical(_tmpkeys[idx], nn.log_softmax(x[0].transpose()).transpose())
                conf = conf.at[idx].set(new_value)
                cache[0] = jnp.roll(cache[0], -1, axis=0)
                cache[0] = cache[0].at[-1].set(nn.one_hot(new_value, self.inputDim))

            return conf

        keys = jax.random.split(key, batchSize+1)
        configs = jax.vmap(generate_sample)(keys[:-1])

        if self.orbit is not None:
            orbitIdx = jax.random.choice(keys[-1], self.orbit.orbit.shape[0], shape=(batchSize,))
            configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s),
                               in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        return configs


class CNNAttention(nn.Module):
    L: int = 4
    kernel_size: Tuple[int] = (2,)
    features: Union[Tuple[int], int] = 8
    inputDim: int = 4
    depth: int = 2
    actFun: callable = nn.elu
    orbit: LatticeSymmetry = None
    logProbFactor: float = 1.  # 1 for POVMs and 0.5 for pure wave functions
    param_dtype: type = jnp.float32
    attention_heads: int = 1

    def setup(self) -> None:
        self.attention_mask = jnp.triu(jnp.ones((self.L, self.L), dtype=bool))
        self.attention_module = nn.SelfAttention(num_heads=self.attention_heads, param_dtype=self.param_dtype)

        self.conv_cells = [CNNCell(features=self.features if i != self.depth - 1 else self.inputDim,
                                   kernel_size=self.kernel_size, kernel_dilation=self.kernel_size[0] ** i,
                                   param_dtype=self.param_dtype, name=f'cnn_cell_{i}')
                           for i in range(self.depth)]

        self.paddings = tuple([self.kernel_size[0]] + [self.kernel_size[0] ** i * (self.kernel_size[0] - 1)
                                                       for i in range(1, self.depth)])
        self.cache_sizes = tuple([(self.kernel_size[0], self.inputDim)] +
                                 [(self.paddings[i] + 1, self.features) for i in range(1, self.depth)])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        def evaluate(x):
            x_oh = jax.nn.one_hot(x, self.inputDim)

            y = self.attention(x_oh)
            y = self.cnn_cell(y)

            return jnp.sum(jax.nn.log_softmax(y) * x_oh)

        if self.orbit is not None:
            # Symmetry case
            x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)
            res = jnp.mean(jnp.exp(jax.vmap(evaluate)(x) / self.logProbFactor), axis=0)

            return self.logProbFactor * jnp.log(res)
        else:
            # No symmetry case
            return evaluate(x)

    def attention(self, x_oh):
        return self.attention_module(x_oh, mask=self.attention_mask)

    def cnn_cell(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x[:-1].reshape(1, -1, self.inputDim)

        for i in range(self.depth):
            x = jnp.pad(x, ((0, 0), (self.paddings[i], 0), (0, 0)))
            x = self.conv_cells[i](x)

        return x[0]

    def sample(self, batchSize: int, key) -> jnp.ndarray:
        """
        This implementation is inspired by 'Fast Generation for Convolutional Autoregressive Models' (arXiv:1704.06001).
        """

        def generate_sample(key):
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=np.int64)

            cache = [jnp.zeros(rf) for rf in self.cache_sizes]

            for idx in range(self.L):
                for i in range(self.depth - 1):
                    x = jnp.copy(cache[i])
                    x = self.conv_cells[i](x)

                    cache[i + 1] = jnp.roll(cache[i + 1], -1, axis=0)
                    cache[i + 1] = cache[i + 1].at[-1].set(x[0])

                x = jnp.copy(cache[self.depth - 1])
                x = self.conv_cells[-1](x)

                new_value = jax.random.categorical(_tmpkeys[idx], nn.log_softmax(x[0].transpose()).transpose())
                conf = conf.at[idx].set(new_value)
                y = self.attention(nn.one_hot(conf, self.inputDim))

                cache[0] = jnp.roll(cache[0], -1, axis=0)
                cache[0] = cache[0].at[-1].set(y[idx])

            return conf

        keys = jax.random.split(key, batchSize + 1)
        configs = jax.vmap(generate_sample)(keys[:-1])

        if self.orbit is not None:
            orbitIdx = jax.random.choice(keys[-1], self.orbit.orbit.shape[0], shape=(batchSize,))
            configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s),
                               in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        return configs


class AFFN(nn.Module):
    """
    Autoregressive implementation of a feed forward neural network.

    This implementation is inspired by 'Solving Statistical Mechanics Using Variational Autoregressive Networks'
    DOI: 10.1103/PhysRevLett.122.080602

    :param L: system size
    :param hiddenSize: number of hidden units per layer
    :param inputDim: dimension of the input (4 for POVMs and 2 for spin 1/2)
    :param depth: number of FFN layers
    :param actFun: activation function to be used at the end of every layer
    :param orbit: LatticeSymmetry object that encodes all symmetry transformations applicable to the system
    :param logProbFactor: exponent of the probability (1 for POVMs and 0.5 for pure wave functions)
    """

    L: int = 4
    hiddenSize: int = 8
    inputDim: int = 4
    depth: int = 2
    actFun: callable = nn.elu
    orbit: LatticeSymmetry = None
    logProbFactor: float = 1.  # 1 for POVMs and 0.5 for pure wave functions

    def setup(self) -> None:
        self.dense_layers = [[nn.Dense(features=(self.L - i)*self.hiddenSize if _ != self.depth - 1 else (self.L - i)*4,
                                       use_bias=True if i == 0 else False)
                             for i in range(self.L)] for _ in range(self.depth)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        def evaluate(x: jnp.ndarray) -> jnp.ndarray:
            x_oh = nn.one_hot(x, self.inputDim)
            return jnp.sum(nn.log_softmax(self.ffn_cell(x_oh)) * x_oh)

        if self.orbit is not None:
            # Symmetry case
            x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)
            res = jnp.mean(jnp.exp(jax.vmap(evaluate)(x) / self.logProbFactor), axis=0)

            return self.logProbFactor * jnp.log(res)
        else:
            # No symmetry case
            return evaluate(x)

    def ffn_cell(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x[:-1].reshape(1, -1, self.inputDim)

        h = [jnp.zeros((self.L, 4 if _ == self.depth - 1 else self.hiddenSize)) for _ in range(self.depth)]
        x = jnp.pad(x, ((0, 0), (1, 0), (0, 0)))
        for idx in range(self.L):
            a = x[0, idx]
            for i in range(self.depth):
                a = self.dense_layers[i][idx](a)
                a = a.reshape(self.L - idx, 4 if i == self.depth - 1 else self.hiddenSize)
                h[i] = h[i].at[idx:].set(h[i][idx:] + a)
                a = self.actFun(h[i][idx])

        return self.actFun(h[-1])

    def sample(self, batchSize: int, key: PRNGKeyArray) -> jnp.ndarray:
        def generate_sample(key: PRNGKeyArray) -> jnp.ndarray:
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=int)

            h = [jnp.zeros((self.L, 4 if _ == self.depth - 1 else self.hiddenSize)) for _ in range(self.depth)]
            a = jnp.zeros(self.inputDim)
            for idx in range(self.L):
                for i in range(self.depth):
                    a = self.dense_layers[i][idx](a)
                    a = a.reshape(self.L - idx, 4 if i == self.depth - 1 else self.hiddenSize)
                    h[i] = h[i].at[idx:].set(h[i][idx:] + a)
                    a = self.actFun(h[i][idx])

                logprobs = jax.nn.log_softmax(a.transpose()).transpose()
                new_site = jax.random.categorical(_tmpkeys[idx], logprobs)
                conf = conf.at[idx].set(new_site)
                a = nn.one_hot(new_site, self.inputDim)

            return conf

        keys = jax.random.split(key, batchSize + 1)
        configs = jax.vmap(generate_sample)(keys[:-1])

        if self.orbit is not None:
            orbitIdx = jax.random.choice(keys[-1], self.orbit.orbit.shape[0], shape=(batchSize,))
            configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s),
                               in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        return configs


class DeepNADE(nn.Module):
    """
    Implementation of a deep Neural Autoregressive Distribution Estimation model.

    This implementation is inspired by 'Neural Autoregressive Distribution Estimation' (arXiv:1605.02226).

    :param L: system size
    :param hiddenSize: number of hidden units per layer
    :param inputDim: dimension of the input (4 for POVMs and 2 for spin 1/2)
    :param depth: number of NADE layers
    :param actFun: activation function to be used at the end of every layer
    :param orbit: LatticeSymmetry object that encodes all symmetry transformations applicable to the system
    :param logProbFactor: exponent of the probability (1 for POVMs and 0.5 for pure wave functions)
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
        self.last_layer = [nn.Dense(features=self.inputDim, use_bias=True) for _ in range(self.L)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        def evaluate(x: jnp.ndarray) -> jnp.ndarray:
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

    def nade_cell(self, x: jnp.ndarray) -> jnp.ndarray:
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

    def sample(self, batchSize: int, key: PRNGKeyArray) -> jnp.ndarray:
        def generate_sample(key: PRNGKeyArray) -> jnp.ndarray:
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


class CNNCell(nn.Module):
    """
    Single layer of a CNN.

    :param features: number of hidden units
    :param kernel_size: size of kernel for convolution
    :param kernel_dilation: dilation of the kernel for convolution
    :param actFun: activation function acting on the output after the convolution
    :param param_dtype: data type for network parameters
    """

    features: int = 8
    kernel_size: int = (2,)
    kernel_dilation: int = 1
    actFun: callable = nn.elu
    param_dtype: type = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, kernel_dilation=self.kernel_dilation,
                    padding='VALID', param_dtype=self.param_dtype)(x)
        x = self.actFun(x)
        return x


class GatedCNNCell(nn.Module):
    """
    Single layer of a gated CNN.

    :param features: number of hidden units
    :param kernel_size: size of kernel for convolution
    :param kernel_dilation: dilation of the kernel for convolution
    :param param_dtype: data type for network parameters
    """

    features: int = 8
    kernel_size: int = (2,)
    kernel_dilation: int = 1
    param_dtype: type = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, kernel_dilation=self.kernel_dilation,
                    padding='VALID', param_dtype=self.param_dtype)(x)

        a, g = jnp.split(x, 2, axis=-1)
        x = nn.sigmoid(g) * jnp.tanh(a)
        return x
