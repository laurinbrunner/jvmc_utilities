import jax.numpy as jnp
import jVMC
import jax
import jvmc_utilities
import pytest


@pytest.mark.slow
def test_POVMCNN():
    L = 4

    cnn = jvmc_utilities.nets.POVMCNN(L=L)

    psi_cnn = jVMC.vqs.NQS(cnn, seed=1234)
    psi_rnn = jVMC.util.util.init_net({"batch_size": 5000, "net1":
        {"type": "RNN", "parameters": {"inputDim": 4, "logProbFactor": 1, "hiddenSize": 6, "L": L, "depth": 2}}},
                                      (L,), 1234)

    sampler_cnn = jVMC.sampler.ExactSampler(psi_cnn, (L,), lDim=4, logProbFactor=1)
    sampler_rnn = jVMC.sampler.ExactSampler(psi_rnn, (L,), lDim=4, logProbFactor=1)
    # sampler = jVMC.sampler.MCSampler(psi, (L,), prngkey, numSamples=2000)

    tdvpEquation_cnn = jVMC.util.tdvp.TDVP(sampler_cnn, rhsPrefactor=-1.,
                                       svdTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=True)
    tdvpEquation_rnn = jVMC.util.tdvp.TDVP(sampler_rnn, rhsPrefactor=-1.,
                                           svdTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=True)

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)
    # stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=1e-3, tol=1E-6)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer_cnn = jvmc_utilities.measurement.Measurement(sampler_cnn, povm)
    measurer_cnn.set_observables(["Sz_i"])
    measurer_rnn = jvmc_utilities.measurement.Measurement(sampler_rnn, povm)
    measurer_rnn.set_observables(["Sz_i"])
    init_cnn = jvmc_utilities.state_init.Initializer(psi_cnn, tdvpEquation_cnn, stepper, lind, measurer=measurer_cnn)
    init_rnn = jvmc_utilities.state_init.Initializer(psi_rnn, tdvpEquation_rnn, stepper, lind, measurer=measurer_rnn)

    init_cnn.initialize(measure_step=-1, steps=100)
    init_rnn.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer_cnn.measure()["Sz_i"], measurer_rnn.measure()["Sz_i"], atol=5E-3)
