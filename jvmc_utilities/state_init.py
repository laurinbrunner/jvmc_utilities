import jax.numpy as jnp
from tqdm import tqdm
from typing import Optional, Union
import jVMC
from . import Measurement


class Initializer:
    def __init__(
            self,
            psi: jVMC.vqs.NQS,
            tdvpEquation: jVMC.util.TDVP,
            stepper: Union[jVMC.util.Euler, jVMC.util.AdaptiveHeun],
            lindbaldian: jVMC.operator.POVMOperator,
            measurer: Optional[Measurement] = None
    ) -> None:
        self.psi = psi
        self.tdvpEquation = tdvpEquation
        self.stepper = stepper
        self.lindbladian = lindbaldian
        self.measurer = measurer

        self.iteration_count = 0
        self.times = []
        self.results = {}

    def initialize_no_measurement(self, steps: int = 300) -> None:
        """
        Calculates time evolution for a given Lindbladian to obtain its steady state after a set number of `steps`.

        This method does not perform measurements and thus runs faster.

        :param steps: Number of time steps.
        """
        for _ in tqdm(range(steps)):
            dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=self.lindbladian,
                                       psi=self.psi)

            self.psi.set_parameters(dp)

    def initialize(self, steps: int = 300) -> None:
        """
        Calculates time evolution for a given Lindbladian to obtain its steady state after a set number of `steps`.

        This method also performs measurements during every time step. The wanted observables must be specified
        beforehand in the Measurement object `measurer`. Measurement results will be stored in the object variable
        `results`.

        :param steps: Number of time steps.
        """
        results = {}

        _res = self.measurer.measure()
        for obs in self.measurer.observables:
            results[obs] = [_res[obs]]
        self.times.append(0)

        for _ in tqdm(range(steps)):
            dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=self.lindbladian,
                                       psi=self.psi)

            self.psi.set_parameters(dp)

            _res = self.measurer.measure()
            for obs in self.measurer.observables:
                results[obs].append(_res[obs])
            self.times.append(self.times[-1] + dt)

        self.results = {}
        for obs in results.keys():
            self.results[obs] = jnp.array(results[obs])
        self.times = jnp.array(self.times)


def copy_state(source: jVMC.vqs.NQS, target: jVMC.vqs.NQS) -> None:
    target.set_parameters(source.get_parameters())
