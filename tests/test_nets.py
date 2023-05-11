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
    init_cnn = jvmc_utilities.time_evolve.Initializer(psi_cnn, tdvpEquation_cnn, stepper, lind, measurer=measurer_cnn)
    init_rnn = jvmc_utilities.time_evolve.Initializer(psi_rnn, tdvpEquation_rnn, stepper, lind, measurer=measurer_rnn)

    init_cnn.initialize(measure_step=-1, steps=100)
    init_rnn.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer_cnn.measure()["Sz_i"], measurer_rnn.measure()["Sz_i"], atol=5E-3)


@pytest.mark.slow
def test_POVMCNNGated():
    L = 4

    cnn = jvmc_utilities.nets.POVMCNNGated(L=L)

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
    init_cnn = jvmc_utilities.time_evolve.Initializer(psi_cnn, tdvpEquation_cnn, stepper, lind, measurer=measurer_cnn)
    init_rnn = jvmc_utilities.time_evolve.Initializer(psi_rnn, tdvpEquation_rnn, stepper, lind, measurer=measurer_rnn)

    init_cnn.initialize(measure_step=-1, steps=100)
    init_rnn.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer_cnn.measure()["Sz_i"], measurer_rnn.measure()["Sz_i"], atol=5E-3)


@pytest.mark.slow
def test_DeepNade():
    L = 4

    nade = jvmc_utilities.nets.DeepNADE(L=L, depth=1)

    psi_nade = jVMC.vqs.NQS(nade, seed=1234)
    psi_rnn = jVMC.util.util.init_net({"batch_size": 5000, "net1":
        {"type": "RNN", "parameters": {"inputDim": 4, "logProbFactor": 1, "hiddenSize": 6, "L": L, "depth": 2}}},
                                      (L,), 1234)

    sampler_nade = jVMC.sampler.ExactSampler(psi_nade, (L,), lDim=4, logProbFactor=1)
    sampler_rnn = jVMC.sampler.ExactSampler(psi_rnn, (L,), lDim=4, logProbFactor=1)
    # sampler = jVMC.sampler.MCSampler(psi, (L,), prngkey, numSamples=2000)

    tdvpEquation_nade = jVMC.util.tdvp.TDVP(sampler_nade, rhsPrefactor=-1.,
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

    measurer_nade = jvmc_utilities.measurement.Measurement(sampler_nade, povm)
    measurer_nade.set_observables(["Sz_i"])
    measurer_rnn = jvmc_utilities.measurement.Measurement(sampler_rnn, povm)
    measurer_rnn.set_observables(["Sz_i"])
    init_nade = jvmc_utilities.time_evolve.Initializer(psi_nade, tdvpEquation_nade, stepper, lind,
                                                       measurer=measurer_nade)
    init_rnn = jvmc_utilities.time_evolve.Initializer(psi_rnn, tdvpEquation_rnn, stepper, lind, measurer=measurer_rnn)

    init_nade.initialize(measure_step=-1, steps=100)
    init_rnn.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer_nade.measure()["Sz_i"], measurer_rnn.measure()["Sz_i"], atol=5E-3)


@pytest.mark.slow
def test_AFFN():
    L = 4

    affn = jvmc_utilities.nets.AFFN(L=L, depth=1)

    psi_affn = jVMC.vqs.NQS(affn, seed=1234)
    psi_rnn = jVMC.util.util.init_net({"batch_size": 5000, "net1":
        {"type": "RNN", "parameters": {"inputDim": 4, "logProbFactor": 1, "hiddenSize": 6, "L": L, "depth": 2}}},
                                      (L,), 1234)

    sampler_affn = jVMC.sampler.ExactSampler(psi_affn, (L,), lDim=4, logProbFactor=1)
    sampler_rnn = jVMC.sampler.ExactSampler(psi_rnn, (L,), lDim=4, logProbFactor=1)
    # sampler = jVMC.sampler.MCSampler(psi, (L,), prngkey, numSamples=2000)

    tdvpEquation_affn = jVMC.util.tdvp.TDVP(sampler_affn, rhsPrefactor=-1.,
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

    measurer_affn = jvmc_utilities.measurement.Measurement(sampler_affn, povm)
    measurer_affn.set_observables(["Sz_i"])
    measurer_rnn = jvmc_utilities.measurement.Measurement(sampler_rnn, povm)
    measurer_rnn.set_observables(["Sz_i"])
    init_affn = jvmc_utilities.time_evolve.Initializer(psi_affn, tdvpEquation_affn, stepper, lind,
                                                       measurer=measurer_affn)
    init_rnn = jvmc_utilities.time_evolve.Initializer(psi_rnn, tdvpEquation_rnn, stepper, lind, measurer=measurer_rnn)

    init_affn.initialize(measure_step=-1, steps=100)
    init_rnn.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer_affn.measure()["Sz_i"], measurer_rnn.measure()["Sz_i"], atol=5E-3)


@pytest.mark.slow
def test_symmetric_POVMCNNGated():
    L = 4

    orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=True, reflection=False)

    cnn = jvmc_utilities.nets.POVMCNNGated(L=L, orbit=orbit)

    psi = jVMC.vqs.NQS(cnn, seed=1234)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    configs = jnp.array([[[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]])

    logprobs = psi(configs)

    assert jnp.unique(jnp.round(logprobs, 12)).shape[0] == 1


@pytest.mark.slow
def test_symmetric_POVMCNN():
    L = 4

    orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=True, reflection=False)
   
    cnn = jvmc_utilities.nets.POVMCNN(L=L, orbit=orbit)

    psi = jVMC.vqs.NQS(cnn, seed=1234)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    configs = jnp.array([[[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]])

    logprobs = psi(configs)

    assert jnp.unique(jnp.round(logprobs, 12)).shape[0] == 1


@pytest.mark.slow
def test_symmetric_DeepNADE():
    L = 4

    orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=True, reflection=False)
    nade = jvmc_utilities.nets.DeepNADE(L=L, orbit=orbit)

    psi = jVMC.vqs.NQS(nade, seed=1234)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    configs = jnp.array([[[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]])

    logprobs = psi(configs)

    assert jnp.unique(jnp.round(logprobs, 12)).shape[0] == 1


@pytest.mark.slow
def test_symmetric_AFFN():
    L = 4

    orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=True, reflection=False)
    affn = jvmc_utilities.nets.AFFN(L=L, orbit=orbit)

    psi = jVMC.vqs.NQS(affn, seed=1234)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    configs = jnp.array([[[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]])

    logprobs = psi(configs)

    assert jnp.unique(jnp.round(logprobs, 12)).shape[0] == 1
