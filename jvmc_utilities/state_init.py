import jax.numpy as jnp
import jVMC
import jvmc_utilities
from tqdm import tqdm


class Initializer:
    def __init__(self, psi, tdvpEquation, stepper, lindbaldian, measurer=None):
        self.psi = psi
        self.tdvpEquation = tdvpEquation
        self.stepper = stepper
        self.lindbladian = lindbaldian
        self.measurer = measurer

        self.iteration_count = 0
        self.times = []
        self.results = {}

    def initialize_no_measurement(self, steps=300):
        """
        Calculates time evolution for a given Lindbladian to obtain its steady state after a set number of `steps`.

        This method does not perform measurements and thus runs faster.

        :param steps: Number of time steps.
        """
        for _ in tqdm(range(steps)):
            dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=self.lindbladian,
                                       psi=self.psi)

            self.psi.set_parameters(dp)

    def initialize(self, steps=300):
        """
        Calculates time evolution for a given Lindbladian to obtain its steady state after a set number of `steps`.

        This method also performs measurements during every time step. The wanted observables must be specified
        beforehand in the Measurement object `measurer`. Measurement results will be stored in the object variable
        `results`.

        :param steps: Number of time steps.
        """
        results = {}
        for obs in self.measurer.observables:
            results[obs] = []

        for _ in tqdm(range(steps)):
            dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=self.lindbladian,
                                       psi=self.psi)

            _res = self.measurer.measure()
            for obs in self.measurer.observables:
                results[obs].append(_res[obs])
            self.times.append(self.times[-1] + dt)

            self.psi.set_parameters(dp)

        self.results = {}
        for obs in results.keys():
            self.results[obs] = jnp.array(results[obs])


if __name__ == '__main__':
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
    jvmc_utilities.operators.initialisation_operators(povm)
    lind = jVMC.operator.POVMOperator(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    init = Initializer(psi, tdvpEquation, stepper, lind)

    init.initialize_no_measurement()