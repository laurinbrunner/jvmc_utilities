import jax.numpy as jnp
import jVMC
import pytest

import jvmc_utilities.state_init


@pytest.fixture(scope='module')
def setup_method():
    L = 4
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
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})
    init = jvmc_utilities.state_init.Initializer(psi, tdvpEquation, stepper, lind)

    init.initialize_no_measurement()

    return sampler, povm


def test_Sz_l(setup_method):
    sampler, povm = setup_method
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sx_l", "Sy_l", "Sz_l", "N"])
    results = measurer.measure()

    assert(jnp.allclose(results["Sx_l"], 0, atol=1E-4))
    assert(jnp.allclose(results["Sy_l"], 0, atol=1E-4))
    assert(jnp.allclose(results["Sz_l"], jnp.array([1, -1, 1, -1]), atol=1E-3))
    assert(jnp.allclose(results["N"], jnp.array([1, 0]), atol=1E-3))


def test_M_sq(setup_method):
    sampler, povm = setup_method
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["M_sq"])
    results = measurer.measure()

    print(results["M_sq"])
    assert (jnp.allclose(results["M_sq"], 1, atol=1E-3))


def test_M_sq_2(setup_method):
    sampler, povm = setup_method
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["M_sq"])
    results = measurer.measure()

    print(results["M_sq"])
    assert (jnp.allclose(results["M_sq"], 1, atol=1E-3))
