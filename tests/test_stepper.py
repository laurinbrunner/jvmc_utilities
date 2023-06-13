import jax.numpy as jnp
import pytest
import jvmc_utilities
import jVMC


def test_BulirschStoer_normal_ODE():
    """
    Test the BulirschStoer class with a simple differntial equation.
    """
    def f(y, t, intStep=-1):
        return y * jnp.sin(t)

    maxStep = 0.3
    stepper = jvmc_utilities.stepper.BulirschStoer(maxStep=maxStep)

    y = jnp.zeros(100)
    t = jnp.zeros(100)
    y0 = 0.4
    y = y.at[0].set(y0)
    for i in range(1, 100):
        y_new, dt = stepper.step(t[i - 1], f, y[i - 1])
        y = y.at[i].set(y_new[0])
        t = t.at[i].set(t[i - 1] + dt)

    assert(jnp.allclose(y, 0.4 * jnp.exp(-jnp.cos(t) + 1)))
    assert(jnp.all(jnp.diff(t) <= maxStep * 1.01))


@pytest.mark.slow
def test_BulirschStoer_CNN():
    """
    Test the BulirschStoer class with a VMC problem and check it against the analytical results.
    """
    L = 2
    cnn = jvmc_utilities.nets.POVMCNN(L=L)
    psi_cnn = jVMC.vqs.NQS(cnn, seed=1234)

    sampler_cnn = jVMC.sampler.ExactSampler(psi_cnn, (L,), lDim=4, logProbFactor=1)

    tdvpEquation_cnn = jVMC.util.tdvp.TDVP(sampler_cnn, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0,
                                           makeReal='real', crossValidation=False)

    init_stepper = jVMC.util.Euler(timeStep=1e-2)
    stepper = jvmc_utilities.stepper.BulirschStoer(timeStep=1e-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})

    measurer_cnn = jvmc_utilities.measurement.Measurement(sampler_cnn, povm)
    measurer_cnn.set_observables(["Sz_i"])
    init_cnn = jvmc_utilities.time_evolve.Initializer(psi_cnn, tdvpEquation_cnn, init_stepper, lind,
                                                      measurer=measurer_cnn)

    init_cnn.initialize(steps=100)

    evol = jvmc_utilities.time_evolve.TimeEvolver(psi_cnn, tdvpEquation_cnn, stepper, measurer_cnn)

    lindbladian = jVMC.operator.POVMOperator(povm)
    lindbladian.add({"name": "decaydown", "strength": 1.0, "sites": (0,)})
    lindbladian.add({"name": "decayup", "strength": 1.0, "sites": (1,)})

    evol.run(lindbladian, 2.0)

    assert(jnp.allclose(evol.results["Sz_i"][:, 0], 2*jnp.exp(-evol.times) - 1, atol=1E-2))
    assert(jnp.allclose(evol.results["Sz_i"][:, 1], 1 - 2*jnp.exp(-evol.times), atol=1E-2))
