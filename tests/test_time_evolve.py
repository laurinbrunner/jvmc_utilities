import jax
import jax.numpy as jnp
import jVMC
import jvmc_utilities.time_evolve
import pytest
from clu import metric_writers


@pytest.fixture(scope='module')
def setup_updown():
    L = 2
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
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind)

    init.initialize(measure_step=-1)
    params = psi.get_parameters()

    return psi, params, sampler, povm, tdvpEquation, stepper


@pytest.mark.slow
def test_with_measurement(setup_updown):
    psi, params, sampler, povm, tdvpEquation, stepper = setup_updown
    # Make sure psi is at start state
    psi.set_parameters(params)

    steps = 200

    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "downdown_dis", "strength": 1.0, "sites": (0, 1)})
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i", "N"])
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(steps=steps)

    rho_t = jnp.zeros((4, 4, steps + 1))
    rho_t = rho_t.at[1, 1].set(jnp.exp(-init.times))
    rho_t = rho_t.at[3, 3].set(1 - jnp.exp(-init.times))

    Z = jnp.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
                   [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]])
    E = jnp.array([jnp.eye(4), jnp.eye(4)])
    assert jnp.allclose(init.results["Sz_i"], jnp.einsum("aij, jit -> ta", Z, rho_t), atol=1E-2)
    assert jnp.allclose(init.results["N"], jnp.einsum("aij, jit -> ta", (Z + E) / 2, rho_t), atol=1E-2)


def test_measurestep(setup_updown):
    psi, params, sampler, povm, tdvpEquation, stepper = setup_updown
    # Make sure psi is at start state
    psi.set_parameters(params)

    steps = 20

    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "downdown_dis", "strength": 1.0, "sites": (0, 1)})
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])

    params = psi.get_parameters()
    init1 = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)
    init2 = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init1.initialize(steps=steps, measure_step=0)
    psi.set_parameters(params)
    init2.initialize(steps=steps, measure_step=1)

    assert init1.results["Sz_i"].shape[0] == 2 * init2.results["Sz_i"].shape[0] - 1
    assert jnp.allclose(init1.times[::2], init2.times)
    assert jnp.allclose(init1.results["Sz_i"][::2, 0], init2.results["Sz_i"][:, 0])


def test_no_measurement_with_convergence():
    L = 2
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
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind)

    init.initialize(convergence=True, atol=1E-2, measure_step=-1)
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])

    assert jnp.allclose(measurer.measure()["Sz_i"], jnp.array([1, -1]), atol=1E-2)


def test_with_measurement_with_convergence(setup_updown):
    psi, params, sampler, povm, tdvpEquation, stepper = setup_updown
    # Make sure psi is at start state
    psi.set_parameters(params)

    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "downdown_dis", "strength": 1.0, "sites": (0, 1)})
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i", "N"])
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer)

    init.initialize(convergence=True, atol=1E-2)
    steps = len(init.times)

    rho_t = jnp.zeros((4, 4, steps))
    rho_t = rho_t.at[1, 1].set(jnp.exp(-init.times))
    rho_t = rho_t.at[3, 3].set(1 - jnp.exp(-init.times))

    Z = jnp.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
                   [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]])
    E = jnp.array([jnp.eye(4), jnp.eye(4)])
    assert jnp.allclose(init.results["Sz_i"], jnp.einsum("aij, jit -> ta", Z, rho_t), atol=1E-2)
    assert jnp.allclose(init.results["N"], jnp.einsum("aij, jit -> ta", (Z + E) / 2, rho_t), atol=1E-2)


@pytest.mark.slow
def test_momentum(setup_updown):
    psi, params, sampler, povm, tdvpEquation, stepper = setup_updown
    # Make sure psi is at start state
    psi.set_parameters(params)

    steps = 200

    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "downdown_dis", "strength": 1.0, "sites": (0, 1)})
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i", "N"])
    init = jvmc_utilities.time_evolve.Initializer(psi, tdvpEquation, stepper, lind, measurer=measurer, momentum=0.9)

    init.initialize(steps=steps)

    rho = jnp.zeros((4, 4))
    rho = rho.at[1, 1].set(0)
    rho = rho.at[3, 3].set(1)

    Z = jnp.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
                   [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]])
    E = jnp.array([jnp.eye(4), jnp.eye(4)])
    assert jnp.allclose(init.results["Sz_i"][-1], jnp.einsum("aij, ji -> a", Z, rho), atol=1E-2)
    assert jnp.allclose(init.results["N"][-1], jnp.einsum("aij, ji -> a", (Z + E) / 2, rho), atol=1E-2)


def test_copy_state(setup_updown):
    psi_source, params, sampler_source, povm, tdvpEquation, stepper = setup_updown
    # Make sure psi is at start state
    psi_source.set_parameters(params)

    L = sampler_source.sampleShape[0]
    psi_target = jVMC.util.util.init_net({"batch_size": 5000, "net1":
        {"type": "RNN", "parameters": {"inputDim": 4, "logProbFactor": 1,
                                       "hiddenSize": 3, "L": L, "depth": 1,
                                       "cell": "RNN"}}},
                                  (L,), 123)
    sampler_target = jVMC.sampler.ExactSampler(psi_target, (L,), lDim=4, logProbFactor=1)

    measurer_source = jvmc_utilities.measurement.Measurement(sampler_source, povm)
    measurer_target = jvmc_utilities.measurement.Measurement(sampler_target, povm)

    measurer_source.set_observables(["Sz_i"])
    measurer_target.set_observables(["Sz_i"])

    jvmc_utilities.copy_state(psi_source, psi_target)

    res_source = measurer_source.measure()
    res_target = measurer_target.measure()

    # Test that both NQS have the same state after the copy process
    assert jnp.allclose(res_source["Sz_i"], res_target["Sz_i"])
    assert jnp.allclose(psi_source.get_parameters(), psi_target.get_parameters())

    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "downdown_dis", "strength": 1.0, "sites": (0, 1)})
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler_source, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0,
                                       makeReal='real', crossValidation=False)

    for _ in range(10):
        dp, dt = stepper.step(0, tdvpEquation, psi_source.get_parameters(), hamiltonian=lind, psi=psi_source)

        psi_source.set_parameters(dp)

    res_target_old = res_target
    res_source = measurer_source.measure()
    res_target = measurer_target.measure()

    # Test that after chaning the source state the target state remains unchanged
    assert not jnp.allclose(res_source["Sz_i"], res_target["Sz_i"])
    assert not jnp.allclose(psi_source.get_parameters(), psi_target.get_parameters())
    assert jnp.allclose(res_target_old["Sz_i"], res_target["Sz_i"])


def test_writer(tmp_path):
    L = 2
    net = jvmc_utilities.nets.DeepNADE(L=L, depth=1, hiddenSize=3, inputDim=4, logProbFactor=1)
    psi = jVMC.vqs.NQS(net, seed=1234, batchSize=5000)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0,
                                       makeReal='real', crossValidation=False)
    stepper = jVMC.util.stepper.Euler(timeStep=1E-2)
    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "decayup", "strength": 5.0, "sites": (0,)})
    lind.add({"name": "decayup", "strength": 5.0, "sites": (1,)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i", "N"])
    writerfile = tmp_path / "tensorboard_test"
    print(str(writerfile))
    writer = metric_writers.summary_writer.SummaryWriter(str(tmp_path / "tensorboard_test"))
    time_evolver = jvmc_utilities.time_evolve.TimeEvolver(psi, tdvpEquation, stepper, measurer, writer=writer,
                                                          additional_hparams={"test": 0},
                                                          parameter_file=tmp_path / "test_parameter")

    time_evolver.run(lind, 0.1001)

    assert (tmp_path / "test_parameter").exists()
    assert "ElocMean" in time_evolver.meta_data.keys()
    assert time_evolver.times.shape == (12,)


def test_timing(tmp_path):
    L = 2
    net = jvmc_utilities.nets.DeepNADE(L=L, depth=1, hiddenSize=3, inputDim=4, logProbFactor=1)
    psi = jVMC.vqs.NQS(net, seed=1234, batchSize=5000)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0,
                                       makeReal='real', crossValidation=False)
    stepper = jVMC.util.stepper.Euler(timeStep=1E-2)
    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "decayup", "strength": 5.0, "sites": (0,)})
    lind.add({"name": "decayup", "strength": 5.0, "sites": (1,)})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i", "N"])
    time_evolver = jvmc_utilities.time_evolve.TimeEvolver(psi, tdvpEquation, stepper, measurer,
                                                          timing_file=tmp_path / "test_timing")

    time_evolver.run(lind, 0.1001)

    assert (tmp_path / "test_timing").exists()
