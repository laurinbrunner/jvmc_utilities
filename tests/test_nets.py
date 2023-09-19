import jax.numpy as jnp
import jVMC
import jax
import jvmc_utilities
import pytest


@pytest.mark.slow
def test_POVMCNN():
    L = 4

    cnn = jvmc_utilities.nets.POVMCNN(L=L, depth=2, features=4)

    psi = jVMC.vqs.NQS(cnn, seed=1234)

    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0, makeReal='real',
                                       crossValidation=False)

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer.measure()["Sz_i"], jnp.array([1., -1., 1., -1.]), atol=0.02)


@pytest.mark.slow
def test_POVMCNNResidual():
    L = 4

    cnn = jvmc_utilities.nets.POVMCNNResidual(L=L, depth=2, features=4)

    psi = jVMC.vqs.NQS(cnn, seed=1234)

    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0, makeReal='real',
                                       crossValidation=False)

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer.measure()["Sz_i"], jnp.array([1., -1., 1., -1.]), atol=0.01)


@pytest.mark.slow
def test_MCMC_CNN():
    L = 4

    cnn = jvmc_utilities.nets.MCMC_CNN(depth=2, features=4, kernel_size=(4,))

    psi = jVMC.vqs.NQS(cnn, seed=1234)

    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0, makeReal='real',
                                       crossValidation=False)

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer.measure()["Sz_i"], jnp.array([1., -1., 1., -1.]), atol=0.01)


@pytest.mark.slow
def test_CNNAttention():
    L = 4

    cnn = jvmc_utilities.nets.CNNAttention(L=L, depth=2, features=4)

    psi = jVMC.vqs.NQS(cnn, seed=1234)

    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0, makeReal='real',
                                       crossValidation=False)

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer.measure()["Sz_i"], jnp.array([1., -1., 1., -1.]), atol=0.01)


@pytest.mark.slow
def test_CNNAttentionResidual():
    L = 4

    cnn = jvmc_utilities.nets.CNNAttentionResidual(L=L, depth=2, features=4)

    psi = jVMC.vqs.NQS(cnn, seed=1234)

    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0, makeReal='real',
                                       crossValidation=False)

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer.measure()["Sz_i"], jnp.array([1., -1., 1., -1.]), atol=0.03)


@pytest.mark.slow
def test_POVMCNNGated():
    L = 4

    cnn = jvmc_utilities.nets.POVMCNNGated(L=L, depth=2, features=4)

    psi = jVMC.vqs.NQS(cnn, seed=1234)

    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0, makeReal='real',
                                       crossValidation=False)

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer.measure()["Sz_i"], jnp.array([1., -1., 1., -1.]), atol=0.02)


@pytest.mark.slow
def test_DeepNade():
    L = 4

    nade = jvmc_utilities.nets.DeepNADE(L=L, depth=1)

    psi = jVMC.vqs.NQS(nade, seed=1234)

    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0, makeReal='real',
                                       crossValidation=False)

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer.measure()["Sz_i"], jnp.array([1., -1., 1., -1.]), atol=0.01)


@pytest.mark.slow
def test_AFFN():
    L = 4

    affn = jvmc_utilities.nets.AFFN(L=L, depth=1)

    psi_affn = jVMC.vqs.NQS(affn, seed=1234)

    sampler_affn = jVMC.sampler.ExactSampler(psi_affn, (L,), lDim=4, logProbFactor=1)

    tdvpEquation_affn = jVMC.util.tdvp.TDVP(sampler_affn, rhsPrefactor=-1.,
                                       pinvTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=False)

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})

    measurer_affn = jvmc_utilities.measurement.Measurement(sampler_affn, povm)
    measurer_affn.set_observables(["Sz_i"])
    init_affn = jvmc_utilities.time_evolve.Initializer(psi_affn, tdvpEquation_affn, stepper, lind,
                                                       measurer=measurer_affn)

    init_affn.initialize(measure_step=-1, steps=100)

    assert jnp.allclose(measurer_affn.measure()["Sz_i"], jnp.array([1., -1., 1., -1.]), atol=0.01)


@pytest.mark.slow
def test_symmetric_POVMCNNGated():
    L = 4

    orbit = jVMC.util.symmetries.get_orbit_1D(L, "translation")

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

    orbit = jVMC.util.symmetries.get_orbit_1D(L, "translation")
   
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
def test_symmetric_POVMCNNResidual():
    L = 4

    orbit = jVMC.util.symmetries.get_orbit_1D(L, "translation")

    cnn = jvmc_utilities.nets.POVMCNNResidual(L=L, orbit=orbit)

    psi = jVMC.vqs.NQS(cnn, seed=1234)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    configs = jnp.array([[[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]])

    logprobs = psi(configs)

    assert jnp.unique(jnp.round(logprobs, 12)).shape[0] == 1


@pytest.mark.slow
def test_symmetric_CNNAttention():
    L = 4

    orbit = jVMC.util.symmetries.get_orbit_1D(L, "translation")

    cnn = jvmc_utilities.nets.CNNAttention(L=L, orbit=orbit)

    psi = jVMC.vqs.NQS(cnn, seed=1234)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    configs = jnp.array([[[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]])

    logprobs = psi(configs)

    assert jnp.unique(jnp.round(logprobs, 12)).shape[0] == 1


@pytest.mark.slow
def test_symmetric_CNNAttentionResidual():
    L = 4

    orbit = jVMC.util.symmetries.get_orbit_1D(L, "translation")

    cnn = jvmc_utilities.nets.CNNAttentionResidual(L=L, orbit=orbit)

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

    orbit = jVMC.util.symmetries.get_orbit_1D(L, "translation")
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

    orbit = jVMC.util.symmetries.get_orbit_1D(L, "translation")
    affn = jvmc_utilities.nets.AFFN(L=L, orbit=orbit)

    psi = jVMC.vqs.NQS(affn, seed=1234)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    configs = jnp.array([[[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]])

    logprobs = psi(configs)

    assert jnp.unique(jnp.round(logprobs, 12)).shape[0] == 1
