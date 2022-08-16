import jax.numpy as jnp
import jVMC
import jvmc_utilities.state_init
import pytest


@pytest.fixture(scope='module')
def setup_updown():
    L = 2
    psi = jVMC.util.util.init_net({"batch_size": 5000, "net1":
        {"type": "RNN", "parameters": {"inputDim": 4, "logProbFactor": 1,
                                       "hiddenSize": 3, "L": L, "depth": 1,
                                       "cell": "RNN"}}},
                                  (L,), 123)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1.,
                                       svdTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=True)
    stepper = jVMC.util.stepper.Euler(timeStep=1E-2)
    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    jvmc_utilities.operators.initialisation_operators(povm)
    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    init = jvmc_utilities.state_init.Initializer(psi, tdvpEquation, stepper, lind)

    init.initialize_no_measurement()

    return psi, sampler, povm


@pytest.mark.slow
def test_with_measurement(setup_updown):
    psi, sampler, povm = setup_updown
    steps = 200

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1.,
                                       svdTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=True)
    stepper = jVMC.util.stepper.Euler(timeStep=1E-2)

    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "downdown_dis", "strength": 1.0, "sites": (0, 1)})
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i", "N"])
    init = jvmc_utilities.state_init.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(steps=steps)

    rho_t = jnp.zeros((4, 4, steps + 1))
    rho_t = rho_t.at[1, 1].set(jnp.exp(-init.times))
    rho_t = rho_t.at[3, 3].set(1 - jnp.exp(-init.times))

    Z = jnp.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
                   [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]])
    E = jnp.array([jnp.eye(4), jnp.eye(4)])
    assert jnp.allclose(init.results["Sz_i"], jnp.einsum("aij, jit -> ta", Z, rho_t), atol=1E-2)
    assert jnp.allclose(init.results["N"], jnp.einsum("aij, jit -> ta", (Z + E) / 2, rho_t), atol=1E-2)
