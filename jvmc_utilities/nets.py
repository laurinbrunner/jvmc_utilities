import jax
import jax.numpy as jnp
import numpy as np
import jVMC
from flax import linen as nn
import jvmc_utilities
import matplotlib.pyplot as plt


class POVMCNN(nn.Module):
    """
    Autoregressive implementation of a Convolutional Neural Network.
    """

    L: int = 4
    kernel_size: int = (2,)
    kernel_dilation: int = 1
    features: int = 8
    inputDim: int = 4
    actFun: callable = nn.elu

    # As of yet unused variables
    hiddenSize: int = 10
    depth: int = 1
    initScale: float = 1.0
    logProbFactor: float = 0.5
    realValuedOutput: bool = False
    realValuedParams: bool = True

    def __call__(self, x):
        x = jax.nn.one_hot(x, self.inputDim)
        probs = jax.nn.log_softmax(self.cnn_cell(x))

        return jnp.sum(probs * x, dtype=np.float64)

    @nn.compact
    def cnn_cell(self, x):
        x = x.reshape(1, -1, self.inputDim)

        x = jnp.pad(x, ((0, 0), (2, 0), (0, 0)))

        x = nn.Conv(features=4, kernel_size=self.kernel_size, kernel_dilation=self.kernel_dilation,
                    padding='VALID')(x[:, :-1, :])  # Last one omitted since it doesn't hold physical meaning ?

        # a, g = jnp.split(x, 2, axis=-1)
        # x = nn.sigmoid(g) * jnp.tanh(a)

        x = self.actFun(x)

        x = jnp.pad(x, ((0, 0), (3, 0), (0, 0)))

        x = nn.Conv(features=4, kernel_size=self.kernel_size, kernel_dilation=2*self.kernel_dilation,
                    padding='VALID')(x[:, :-1, :])

        x = self.actFun(x)

        # a, g = jnp.split(x, 2, axis=-1)
        # x = nn.sigmoid(g) * jnp.tanh(a)

        return x[0]

    def sample(self, batchSize, key):
        def generate_sample(key):
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=np.int64)
            conf_oh = jax.nn.one_hot(conf, self.inputDim)
            for idx in range(self.L):
                logprobs = jax.nn.log_softmax(self.cnn_cell(conf_oh)[idx].transpose()).transpose()
                conf = conf.at[idx].set(jax.random.categorical(_tmpkeys[idx], logprobs))
            return conf

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)


if __name__ == '__main__':
    L = 4
    prngkey = jax.random.PRNGKey(0)

    cnn = POVMCNN(L=L)

    psi = jVMC.vqs.NQS(cnn, seed=1234)

    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)
    #sampler = jVMC.sampler.MCSampler(psi, (L,), prngkey, numSamples=2000)

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1.,
                                       svdTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=True)

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)
    #stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=1e-3, tol=1E-6)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])
    init = jvmc_utilities.state_init.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(100)

    for i in range(L):
        plt.plot(init.times, init.results["Sz_i"][:, i], label=f"{i}")
    plt.legend()
    plt.show()
