import jax
import jax.numpy as jnp
import jVMC
from flax import linen as nn
import jvmc_utilities


class POVMCNN(nn.Module):
    """
    Autoregressive implementation of a Convolutional Neural Network.
    """

    L: int = 4
    kernel_size: int = (2,)
    kernel_dilation: int = None
    features: int = 8
    inputDim: int = 4

    # As of yet unused variables
    hiddenSize: int = 10
    depth: int = 1
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5
    realValuedOutput: bool = False
    realValuedParams: bool = True

    def __call__(self, x):
        probs = self.cnn_cell(x)

        return jnp.sum(probs)

    @nn.compact
    def cnn_cell(self, x):
        x = x.reshape(1, self.L, 1)

        x = jnp.pad(x, ((0, 0), (2, 0), (0, 0)))

        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, kernel_dilation=self.kernel_dilation,
                    padding='VALID')(x[:, :-1, :])  # Last one omitted since it doesn't hold physical meaning ?

        return jnp.sum(x, axis=2)[0]

    def sample(self, batchSize, key):
        def generate_sample(key):
            _tmpkeys = jax.random.split(key, self.L)
            conf = jnp.zeros(self.L, dtype=int)
            for idx in range(self.L):
                logprobs = self.cnn_cell(conf)
                conf = conf.at[idx].set(jax.random.categorical(_tmpkeys[idx], logprobs))
            return conf

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)


if __name__ == '__main__':
    prngkey = jax.random.PRNGKey(0)

    c = POVMCNN()

    psi = jVMC.vqs.NQS(c, seed=1234)
    sampler = jVMC.sampler.MCSampler(psi, (4,), prngkey, numSamples=2000)

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1.,
                                       svdTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=True)

    stepper = jVMC.util.stepper.Euler(timeStep=1E-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": 4})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])
    init = jvmc_utilities.state_init.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    results = init.initialize(200)
