import jax.numpy as jnp
import jVMC

import jvmc_utilities.state_init
from jvmc_utilities import *


def test_Sz_l():
    psi = jVMC.util.util.init_net({"batch_size": 5000, "net1":
        {"type": "RNN", "parameters": {"inputDim": 4, "logProbFactor": 1,
                                       "hiddenSize": 3, "L": 2, "depth": 1,
                                       "cell": "RNN"}}},
                                  (2,), 123)
    sampler = jVMC.sampler.ExactSampler(psi, (2,), lDim=4, logProbFactor=1)
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1.,
                                       svdTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=True)
    stepper = jVMC.util.stepper.Euler(timeStep=1E-2)
    povm = jVMC.operator.POVM({"dim": "1D", "L": 2})
    initialisation_operators(povm)
    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    init = jvmc_utilities.state_init.Initializer(psi, tdvpEquation, stepper, lind)

    init.initialize_no_measurement()

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sx_l", "Sy_l", "Sz_l", "N"])
    results = measurer.measure()

    assert(jnp.allclose(results["Sx_l"], 0, atol=1E-4))
    assert(jnp.allclose(results["Sy_l"], 0, atol=1E-4))
    assert(jnp.allclose(results["Sz_l"], jnp.array([1, -1]), atol=1E-4))
    assert(jnp.allclose(results["N"], jnp.array([1, 0]), atol=1E-4))
