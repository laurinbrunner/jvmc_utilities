import jax.numpy as jnp
from tqdm import tqdm
from typing import Optional, Union
import jVMC
from . import Measurement
import warnings


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
        warnings.warn("initialize_no_measurement method is deprecated. Using initialize(measurestep=-1) instead is "
                      "adviced.", DeprecationWarning)
        self.initialize(measurestep=-1, steps=steps)

    def initialize(self, measurestep: int = 0, steps: int = 300) -> None:
        """
        Calculates time evolution for a given Lindbladian to obtain its steady state after a set number of `steps`.

        This method also performs measurements during every time step. The wanted observables must be specified
        beforehand in the Measurement object `measurer`. Measurement results will be stored in the object variable
        `results`.

        :param measurestep: Number of steps between measurements. If negative no measurements are performed at all.
        :param steps: Number of time steps.

        :raises: ValueError
        """
        if measurestep >= 0:
            if self.measurer is None:
                raise ValueError(f"Trying to measure every {measurestep} steps while no measurer has been defined for"
                                 f"this initializer.")
            self.__with_measurement(measurestep=measurestep, steps=steps)
        else:
            self.__no_measurements(steps=steps)

    def __no_measurements(self, steps: int) -> None:
        """
        Helper function for initialisation without any measurements. Not intended to be called directly.
        """
        for _ in tqdm(range(steps)):
            dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=self.lindbladian,
                                       psi=self.psi)

            self.psi.set_parameters(dp)

    def __with_measurement(self, measurestep: int, steps: int) -> None:
        """
        Helper function for initialisation with measurements. Not intended to be called directly.
        """
        results = {}
        times = []

        # Do measurement on the first state
        _res = self.measurer.measure()
        for obs in self.measurer.observables:
            results[obs] = [_res[obs]]
        if len(self.times) == 0:
            times.append(0)
        else:
            times.append(self.times[-1])

        t = times[-1]

        measurecounter = 0
        for _ in tqdm(range(steps)):
            dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=self.lindbladian,
                                       psi=self.psi)

            t += dt
            self.psi.set_parameters(dp)

            if measurecounter == measurestep:
                _res = self.measurer.measure()
                for obs in self.measurer.observables:
                    results[obs].append(_res[obs])
                times.append(t)
                measurecounter = 0
            else:
                measurecounter += 1

        if len(self.results.keys()) == 0:
            self.results = {}
            for obs in results.keys():
                self.results[obs] = jnp.array(results[obs])
            self.times = jnp.array(times)
        else:
            for obs in results.keys():
                if obs in self.results.keys():
                    self.results[obs] = jnp.concatenate([self.results[obs], jnp.array(results[obs])])
                else:
                    self.results[obs] = jnp.array(results[obs])
            self.times = jnp.concatenate([self.times, jnp.array(times)])


def copy_state(source: jVMC.vqs.NQS, target: jVMC.vqs.NQS) -> None:
    target.set_parameters(source.get_parameters())
