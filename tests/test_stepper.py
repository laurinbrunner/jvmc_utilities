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


def test_minAdaptiveHeun_normal_ODE():
    """
    Test the minAdaptiveHeun class with a simple differntial equation.
    """
    def f(y, t, intStep=-1):
        return y * jnp.sin(t)

    maxStep = 0.3
    stepper = jvmc_utilities.stepper.minAdaptiveHeun(maxStep=maxStep, minStep=1E-3)

    y = jnp.zeros(100)
    t = jnp.zeros(100)
    y0 = 0.4
    y = y.at[0].set(y0)
    for i in range(1, 100):
        y_new, dt = stepper.step(t[i - 1], f, y[i - 1])
        y = y.at[i].set(y_new)
        t = t.at[i].set(t[i - 1] + dt)

    assert(jnp.allclose(y, 0.4 * jnp.exp(-jnp.cos(t) + 1)))
    assert(jnp.all(jnp.diff(t) <= maxStep * 1.01))


@pytest.mark.slow
def test_BulirschStoer_NADE():
    """
    Test the BulirschStoer class with a VMC problem and check it against the analytical results.
    """
    L = 2
    nade = jvmc_utilities.nets.DeepNADE(L=L, depth=1, hiddenSize=4)
    psi = jVMC.vqs.NQS(nade, seed=1234)

    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1., pinvTol=1e-6, diagonalShift=0,
                                           makeReal='real', crossValidation=False)

    init_stepper = jVMC.util.Euler(timeStep=1e-2)
    stepper = jvmc_utilities.stepper.BulirschStoer(timeStep=1e-2)

    povm = jVMC.operator.POVM({"dim": "1D", "L": L})

    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])
    psi.set_parameters(jnp.array([1.362992316347582333, 1.309718333161965065, -4.120035186913138525e-01,
                                       -3.572472984633650595e-01, -4.789574146270756949e-01, -3.352125734090972281e-02,
                                       6.644605100154894750e-02, -2.334889918565755951e-01, -9.599974751472480083e-02,
                                       -3.716400265693666216e-01, -7.278644442558287464e-01, -1.016090869903564453e+00,
                                       6.546531915664672852e-01, -3.570948541164398193e-01, -7.138211131095886230e-01,
                                       -2.590229511260986328e-01, 3.579644560813903809e-01, -7.595479488372802734e-02,
                                       9.080262184143066406e-01, -6.815330684185028076e-02, 9.463173483863079882e-01,
                                       1.001119523273834799e+00, -4.821753233258964100e-01, -4.996536405395142488e-02,
                                       3.851875483151088186e-01, 6.018877913173378280e-01, -4.106500509121587261e-01,
                                       3.319551579064307367e-03, 7.484250987819082157e-01, -1.590429731563470228e-01,
                                       3.754281778670838250e-04, 3.273531457368546449e-01, -9.906406282754677550e-02,
                                       3.341157615199786268e-01, -3.651741881705981019e-01, -1.366536040836884187e-02,
                                       6.806583136495518715e-01, -1.338113323920934843e-02, -3.438223648609659011e-01,
                                       -2.552571975297990292e-01, 2.109783691373451064e-01, -2.187842543661675243e-01,
                                       3.961084881898995769e-01, 2.005230183597154769e-01, -1.508471044948933304e-01,
                                       4.151115837415412235e-01, -3.563525334300171044e-01, -1.123229501359892038e-02,
                                       -3.626062993226688569e-01, 1.503377456893088271e-01, -1.045237826545491328e-01,
                                       7.311906958586402716e-01, -3.074424608339525467e-01, 5.964012321333136413e-01,
                                       -4.422889193256083762e-01, -6.013560631643224408e-01, -4.275487771815209559e-01,
                                       6.642234068422780968e-01, 4.848261378986392645e-01, 7.336878530045323199e-01,
                                       -1.046955138900855342e+00, 1.066234584133627816e+00, 1.082588852946674285e+00,
                                       8.048561929872256604e-01, -6.624207992131688760e-01, 7.613331569367430829e-01,
                                       6.385919156497305016e-01, 1.299773719301246633e+00, 1.498311880522255612e-01,
                                       -8.201827394034544305e-02, -8.231325427672584460e-01, 1.283193739341286777e-01,
                                       -1.541441201528487848e-01, -9.368423052836575282e-01, -3.891042373348477801e-01,
                                       3.177869534370861282e-01]))

    evol = jvmc_utilities.time_evolve.TimeEvolver(psi, tdvpEquation, stepper, measurer)

    lindbladian = jVMC.operator.POVMOperator(povm)
    lindbladian.add({"name": "decaydown", "strength": 1.0, "sites": (0,)})
    lindbladian.add({"name": "decayup", "strength": 1.0, "sites": (1,)})

    evol.run(lindbladian, 2.0)

    assert(jnp.allclose(evol.results["Sz_i"][:, 0], 2*jnp.exp(-evol.times) - 1, atol=1E-2))
    assert(jnp.allclose(evol.results["Sz_i"][:, 1], 1 - 2*jnp.exp(-evol.times), atol=1E-2))
