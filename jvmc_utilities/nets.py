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
            conf = jnp.zeros(self.L, dtype=np.uint8)

            # This list caches the input to the i-th cnn layer
            cache = [jnp.zeros(rf, dtype=self.param_dtype) for rf in self.cache_sizes]

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


class POVMCNNEmbedded(nn.Module):
    """
    Autoregressive implementation of a Convolutional Neural Network where the physical sites get a embedded
    dimensionality.

    This implementation is inspired by 'WaveNet: A Generative Model for Raw Audio' by van den Oord et. al.
    (arXiv:1609.03499).

    :param L: system size
    :param kernel_size: size of kernel for convolution
    :param features: number of hidden units
    :param inputDim: dimension of the input (4 for POVMs and 2 for spin 1/2)
    :param depth: number of convolutional layers
    :param actFun: activation function to be used at the end of every layer
    :param embeddingDimFac: factor by which inputDim is multiplied to get the embedding dimension
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
    embeddingDimFac: int = 1
    orbit: LatticeSymmetry = None
    logProbFactor: float = 1.  # 1 for POVMs and 0.5 for pure wave functions
    param_dtype: type = jnp.float32

    def setup(self) -> None:
        features = self.features
        self.embeddingDim = self.inputDim * self.embeddingDimFac
        if type(self.features) is int:
            features = tuple([self.features] * self.depth)

        self.embedding_cell = nn.Conv(features=self.embeddingDim, kernel_size=(2,), padding='VALID', strides=2,
                                      use_bias=True, param_dtype=self.param_dtype, name="Embedding_Conv")

        self.conv_cells = [CNNCell(features=features[i] if i != self.depth - 1 else self.inputDim**2,
                                   kernel_size=self.kernel_size, actFun=self.actFun, param_dtype=self.param_dtype,
                                   kernel_dilation=self.kernel_size[0]**i)
                           for i in range(self.depth)]

        self.paddings = tuple([self.kernel_size[0]] +
                              [self.kernel_size[0]**i * (self.kernel_size[0] - 1) for i in range(1, self.depth)])

        self.cache_sizes = tuple([(2, self.inputDim), (self.kernel_size[0], self.embeddingDim)] +
                                 [(self.paddings[i] + 1, features[i - 1]) for i in range(1, self.depth)])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        def evaluate(x: jnp.ndarray):
            x_oh = jax.nn.one_hot(x, self.inputDim)
            x_oh2 = jax.nn.one_hot(self.inputDim * x[::2] + x[1::2], self.inputDim**2)

            x_emb = self.actFun(self.embedding_cell(x_oh))

            return jnp.sum(jax.nn.log_softmax(self.cnn_cell(x_emb)) * x_oh2)

        if self.orbit is not None:
            # Symmetry case
            x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)

            res = jnp.mean(jnp.exp(jax.vmap(evaluate)(x) / self.logProbFactor), axis=0)

            return self.logProbFactor * jnp.log(res)
        else:
            # No symmetry case
            return evaluate(x)

    def cnn_cell(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x[:-1].reshape(1, -1, self.embeddingDim)

        for i in range(self.depth):
            x = jnp.pad(x, ((0, 0), (self.paddings[i], 0), (0, 0)))
            x = self.conv_cells[i](x)

        return x[0]

    def sample(self, batchSize: int, key: jax.random.PRNGKeyArray) -> jnp.ndarray:
        """
        This implementation is inspired by 'Fast Generation for Convolutional Autoregressive Models' (arXiv:1704.06001).
        """
        def generate_sample(key: jax.random.PRNGKeyArray) -> jnp.ndarray:
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(2*self.L, dtype=np.uint8)

            # This list caches the input to the i-th cnn layer
            cache = [jnp.zeros(rf, dtype=self.param_dtype) for rf in self.cache_sizes]

            for idx in range(self.L):
                x = self.embedding_cell(cache[0])
                x = self.actFun(x)
                cache[1] = jnp.roll(cache[1], -1, axis=0)
                cache[1] = cache[1].at[-1].set(x[0])

                for i in range(self.depth):
                    x = self.conv_cells[i](cache[i+1])

                    if i != self.depth - 1:
                        cache[i+2] = jnp.roll(cache[i+2], -1, axis=0)
                        cache[i+2] = cache[i+2].at[-1].set(x[0])

                new_value = jax.random.categorical(_tmpkeys[idx], nn.log_softmax(x[0].transpose()).transpose())

                new_value = jnp.array([new_value // self.inputDim, new_value % self.inputDim])
                conf = conf.at[2*idx:2*idx+2].set(new_value)
                cache[0] = nn.one_hot(new_value, self.inputDim)

            return conf

        keys = jax.random.split(key, batchSize+1)
        configs = jax.vmap(generate_sample)(keys[:-1])

        if self.orbit is not None:
            orbitIdx = jax.random.choice(keys[-1], self.orbit.orbit.shape[0], shape=(batchSize,))
            configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s),
                               in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        return configs


class POVMCNNEmbeddedResidual(POVMCNNEmbedded):

    def setup(self) -> None:
        super().setup()
        features = self.features
        if type(self.features) is int:
            features = tuple([self.features] * self.depth)
        self.residual_convs = [nn.Conv(features=features[i] if i != self.depth - 1 else self.inputDim ** 2,
                                       kernel_size=(1,), param_dtype=self.param_dtype)
                               for i in range(self.depth)]

    def cnn_cell(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(1, -1, self.embeddingDim)[:, :-1]

        for i in range(self.depth):
            x_padded = jnp.pad(x, ((0, 0), (self.paddings[i], 0), (0, 0)))
            x = self.actFun(self.conv_cells[i](x_padded) + self.residual_convs[i](
                x if i != 0 else x_padded[:, (self.kernel_size[0] - 1):]))

        return x[0]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        def evaluate(x: jnp.ndarray):
            x_oh = jax.nn.one_hot(x, self.inputDim)
            x_oh2 = jax.nn.one_hot(self.inputDim * x[::2] + x[1::2], self.inputDim ** 2)
            x_emb = self.actFun(self.embedding_cell(x_oh))
            return jnp.sum(jax.nn.log_softmax(self.cnn_cell(x_emb)) * x_oh2)

        if self.orbit is not None:
            # Symmetry case
            x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)

            res = jnp.mean(jnp.exp(jax.vmap(evaluate)(x) / self.logProbFactor), axis=0)

            return self.logProbFactor * jnp.log(res)
        else:
            # No symmetry case
            return evaluate(x)

    def sample(self, batchSize: int, key: jax.random.PRNGKeyArray) -> jnp.ndarray:
        """
        This implementation is inspired by 'Fast Generation for Convolutional Autoregressive Models' (arXiv:1704.06001).
        """

        def generate_sample(key: jax.random.PRNGKeyArray) -> jnp.ndarray:
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(2 * self.L, dtype=np.uint8)

            # This list caches the input to the i-th cnn layer
            cache = [jnp.zeros(rf, dtype=self.param_dtype) for rf in self.cache_sizes]

            for idx in range(self.L):
                x = self.embedding_cell(cache[0])
                x = self.actFun(x)
                cache[1] = jnp.roll(cache[1], -1, axis=0)
                cache[1] = cache[1].at[-1].set(x[0])

                for i in range(self.depth):
                    x = jnp.copy(cache[i + 1])
                    x = self.actFun(self.conv_cells[i](x) + self.residual_convs[i](x[-1].reshape(1, -1)))

                    if i != self.depth - 1:
                        cache[i + 2] = jnp.roll(cache[i + 2], -1, axis=0)
                        cache[i + 2] = cache[i + 2].at[-1].set(x[0])

                new_value = jax.random.categorical(_tmpkeys[idx], nn.log_softmax(x[0].transpose()).transpose())
                new_value = jnp.array([new_value // self.inputDim, new_value % self.inputDim])

                conf = conf.at[2 * idx:2 * idx + 2].set(new_value)
                cache[0] = nn.one_hot(new_value, self.inputDim)

            return conf

        keys = jax.random.split(key, batchSize + 1)
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
        x = x.reshape(1, -1, self.inputDim)[:, :-1]

        for i in range(self.depth):
            x_padded = jnp.pad(x, ((0, 0), (self.paddings[i], 0), (0, 0)))
            x = self.actFun(self.conv_cells[i](x_padded) + self.residual_convs[i](x if i != 0 else x_padded[:, (self.kernel_size[0]-1):]))

        return x[0]

    def sample(self, batchSize: int, key: PRNGKeyArray) -> jnp.ndarray:
        """
        This implementation is inspired by 'Fast Generation for Convolutional Autoregressive Models' (arXiv:1704.06001).
        """
        def generate_sample(key: PRNGKeyArray) -> jnp.ndarray:
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=np.uint8)

            # This list caches the input to the i-th cnn layer
            cache = [jnp.zeros(rf, dtype=self.param_dtype) for rf in self.cache_sizes]

            for idx in range(self.L):
                for i in range(self.depth):
                    x = jnp.copy(cache[i])
                    x = self.actFun(self.conv_cells[i](x) + self.residual_convs[i](x[-1].reshape(1, -1)))

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
        self.attention_mask = jnp.tril(jnp.ones((self.L, self.L), dtype=bool))
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

    def sample(self, batchSize: int, key: PRNGKeyArray) -> jnp.ndarray:
        """
        This implementation is inspired by 'Fast Generation for Convolutional Autoregressive Models' (arXiv:1704.06001).
        """

        def generate_sample(key: PRNGKeyArray):
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=np.uint8)

            cache = [jnp.zeros(rf, dtype=self.param_dtype) for rf in self.cache_sizes]

            for idx in range(self.L):
                for i in range(self.depth):
                    x = jnp.copy(cache[i])
                    x = self.conv_cells[i](x)

                    if i != self.depth - 1:
                        cache[i + 1] = jnp.roll(cache[i + 1], -1, axis=0)
                        cache[i + 1] = cache[i + 1].at[-1].set(x[0])

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


class CNNAttentionResidual(CNNAttention):

    def setup(self) -> None:
        super().setup()
        features = self.features
        if type(self.features) is int:
            features = tuple([self.features] * self.depth)
        self.residual_convs = [nn.Conv(features=features[i] if i != self.depth - 1 else self.inputDim,
                                       kernel_size=(1,), param_dtype=self.param_dtype)
                               for i in range(self.depth)]

    def cnn_cell(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(1, -1, self.inputDim)[:, :-1]

        for i in range(self.depth):
            x_padded = jnp.pad(x, ((0, 0), (self.paddings[i], 0), (0, 0)))
            x = self.actFun(self.conv_cells[i](x_padded) + self.residual_convs[i](x if i != 0 else x_padded[:, (self.kernel_size[0] - 1):]))

        return x[0]

    def sample(self, batchSize: int, key: PRNGKeyArray) -> jnp.ndarray:
        """
        This implementation is inspired by 'Fast Generation for Convolutional Autoregressive Models' (arXiv:1704.06001).
        """

        def generate_sample(key: PRNGKeyArray):
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=np.uint8)

            cache = [jnp.zeros(rf, dtype=self.param_dtype) for rf in self.cache_sizes]

            for idx in range(self.L):
                for i in range(self.depth):
                    x = jnp.copy(cache[i])
                    x = self.actFun(self.conv_cells[i](x) + self.residual_convs[i](x[-1].reshape(1, -1)))

                    if i != self.depth - 1:
                        cache[i + 1] = jnp.roll(cache[i + 1], -1, axis=0)
                        cache[i + 1] = cache[i + 1].at[-1].set(x[0])

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
    param_dtype: type = jnp.float32

    def setup(self) -> None:
        self.dense_layers = [[nn.Dense(features=(self.L - i)*self.hiddenSize if _ != self.depth - 1 else (self.L - i)*4,
                                       use_bias=True if i == 0 else False, param_dtype=self.param_dtype)
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

        h = [jnp.zeros((self.L, 4 if _ == self.depth - 1 else self.hiddenSize), dtype=self.param_dtype)
             for _ in range(self.depth)]
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
            conf = jnp.zeros(self.L, dtype=np.uint8)

            h = [jnp.zeros((self.L, 4 if _ == self.depth - 1 else self.hiddenSize), dtype=self.param_dtype)
                 for _ in range(self.depth)]
            a = jnp.zeros(self.inputDim, dtype=self.param_dtype)
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
    param_dtype: type = jnp.float32

    def setup(self) -> None:
        self.deep_layers = [[nn.Dense(features=self.hiddenSize, use_bias=True if (i == 0 and _ == 0) else False,
                                      param_dtype=self.param_dtype)
                             for i in range(self.depth)] for _ in range(self.L)]
        self.last_layer = [nn.Dense(features=self.inputDim, use_bias=True, param_dtype=self.param_dtype)
                           for _ in range(self.L)]

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
        p = jnp.zeros_like(x, dtype=self.param_dtype)
        x = x[:-1].reshape(1, -1, self.inputDim)

        x = jnp.pad(x, ((0, 0), (1, 0), (0, 0)))
        a = jnp.zeros((self.depth, self.hiddenSize), dtype=self.param_dtype)
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

            conf = jnp.zeros(self.L, dtype=np.uint8)
            a = jnp.zeros((self.depth, self.hiddenSize), dtype=self.param_dtype)
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


class MCMC_CNN(nn.Module):
    """
    Convolutional Neural Network for POVM in the MCMC framework.

    :param inputDim: dimension of the input space
    :param kernel_size: size of the convolutional kernel
    :param features: number of features in the convolutional layer
    :param prefeatures: number of features in the pre-convolutional layer
    :param depth: number of convolutional layers
    :param actFun: activation function
    :param use_bias: whether to use bias in the convolutional layers
    :param param_dtype: datatype of the parameters
    """
    inputDim: int = 4
    kernel_size: Tuple[int] = (2,)
    features: int = 8
    prefeatures: int = 16
    depth: int = 2
    actFun: callable = nn.elu
    use_bias: bool = True
    param_dtype: type = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_oh = nn.one_hot(x, self.inputDim)

        x_oh = nn.Conv(features=self.prefeatures, kernel_size=(2,), padding='CIRCULAR', strides=2,
                       use_bias=self.use_bias, param_dtype=self.param_dtype, name="Pre_Conv")(x_oh)

        for j in range(self.depth):
            x_oh = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding='CIRCULAR',
                           use_bias=self.use_bias, name=f"main Conv {j}")(x_oh)
            x_oh = self.actFun(x_oh)

        return jnp.sum(x_oh)


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


class MCMC_ResNet(nn.Module):
    """
    Residual convolutional block neural network for POVM in the MCMC framework.

    :param prefeatures: Number of features for first convolution to go from computational sites to physical sites
    :param features: Number of features for main convolutions
    :param depth: Number of blocks
    :param bias: Whether to use biases
    :param inputDim: Local dimension of input. 4 for POVM
    :param actFun: Activation function to be used between convolutions. Default is nn.gelu
    :param kernel_size: Size of convolutional filter
    :param param_dtype: Data type for parameters and network output
    """
    prefeatures: int = 16
    features: int = 8
    kernel_size: Tuple[int] = (4,)
    depth: int = 1
    bias: bool = True
    inputDim: int = 4
    actFun: callable = nn.gelu
    param_dtype: type = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        nsites = x.size

        x_oh = nn.one_hot(x, self.inputDim)

        x = nn.Conv(
            features=self.prefeatures,
            kernel_size=(2,),
            padding='CIRCULAR',
            strides=2,
            use_bias=self.bias,
            param_dtype=self.param_dtype,
            kernel_init=jax.nn.initializers.lecun_normal(),
            bias_init=jax.nn.initializers.zeros,
            name="Pre_Conv"
        )(x_oh)

        x = self.actFun(x)

        for nblock in range(self.depth):

            residual = x
            x /= np.sqrt(nblock+1, dtype=self.param_dtype)

            if nblock == 0:
                x /= np.sqrt(2, dtype=self.param_dtype)
            else:
                x = self.actFun(x)

            x = nn.Conv(
                features=self.features,
                kernel_size=self.kernel_size,
                padding="CIRCULAR",
                use_bias=self.bias,
                param_dtype=self.param_dtype,
                dtype=self.param_dtype,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros,
                name=f"block {nblock} first conv"
                )(x)

            x = self.actFun(x)

            x = nn.Conv(
                features=self.features,
                kernel_size=self.kernel_size,
                padding="CIRCULAR",
                use_bias=self.bias and (nblock != self.depth-1),
                param_dtype=self.param_dtype,
                dtype=self.param_dtype,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros,
                name=f"block {nblock} second conv"
                )(x)

            x += residual.mean(axis=-1).reshape(-1, 1)

        x /= np.sqrt(nblock+1, dtype=self.param_dtype)

        x = jax.scipy.special.logsumexp(x) - jnp.log(self.features * nsites)

        return x


def propose_POVM_outcome(key, s, info):
    key, subkey = jax.random.split(key)
    idx = jax.random.randint(subkey, (1,), 0, s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    return s.at[idx].set(jax.random.randint(key, (1,), 0, 4)[0])
