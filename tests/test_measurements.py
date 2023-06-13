import jax.numpy as jnp
import jVMC
import pytest
import jax

import jvmc_utilities


@pytest.fixture(scope='module')
def setup_method():
    L = 4
    psi = jVMC.util.util.init_net({"batch_size": 5000, "net1":
                                  {"type": "RNN", "parameters": {"inputDim": 4, "logProbFactor": 1,
                                                                 "hiddenSize": 3, "L": L, "depth": 1,
                                                                 "cell": "RNN"}}},
                                  (L,), 123)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0,
                                       makeReal='real', crossValidation=False)
    stepper = jVMC.util.stepper.Euler(timeStep=1E-2)
    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    jvmc_utilities.operators.initialisation_operators(povm)
    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind)

    init.initialize(measure_step=-1)

    return sampler, povm


def test_Sz_l(setup_method):
    sampler, povm = setup_method
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sx_i", "Sy_i", "Sz_i", "N"])
    results = measurer.measure()

    assert(jnp.allclose(results["Sx_i"], 0, atol=1E-4))
    assert(jnp.allclose(results["Sy_i"], 0, atol=1E-4))
    assert(jnp.allclose(results["Sz_i"], jnp.array([1, -1, 1, -1]), atol=1E-3))
    assert(jnp.allclose(results["N"], jnp.array([1, 0]), atol=1E-3))


def test_M_sq(setup_method):
    sampler, povm = setup_method
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["M_sq"])
    results = measurer.measure()

    assert (jnp.allclose(results["M_sq"], 1, atol=1E-3))


def test_M_sq_2(setup_method):
    sampler, povm = setup_method
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["M_sq"])
    results = measurer.measure()

    assert (jnp.allclose(results["M_sq"], 1, atol=1E-3))


def test_MC_error():
    L = 4
    psi = jVMC.util.util.init_net({"batch_size": 5000, "net1":
        {"type": "RNN", "parameters": {"inputDim": 4, "logProbFactor": 1,
                                       "hiddenSize": 3, "L": L, "depth": 1,
                                       "cell": "RNN"}}},
                                  (L,), 123)
    sampler = jVMC.sampler.MCSampler(psi, (L,), jax.random.PRNGKey(0))
    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm, mc_errors=True)
    measurer_without = jvmc_utilities.measurement.Measurement(sampler, povm, mc_errors=False)
    measurer.set_observables(["Sz_i", "Sx_i", "Sy_i", "N"])
    measurer_without.set_observables(["Sz_i", "Sx_i", "Sy_i", "N"])

    result_with = measurer.measure()
    result_without = measurer_without.measure()

    assert "Sz_i_MC_error" in result_with.keys()
    assert "Sx_i_MC_error" in result_with.keys()
    assert "Sy_i_MC_error" in result_with.keys()
    assert "N_MC_error" in result_with.keys()
    assert "Sz_i_MC_error" not in result_without.keys()
    assert "Sx_i_MC_error" not in result_without.keys()
    assert "Sy_i_MC_error" not in result_without.keys()
    assert "N_MC_error" not in result_without.keys()
