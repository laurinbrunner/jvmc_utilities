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
            measurer: Optional[Measurement] = None,
            sampler: Optional[Union[jVMC.sampler.MCSampler, jVMC.sampler.ExactSampler]] = None,
            povm: Optional[jVMC.operator.POVM] = None
    ) -> None:
        self.psi = psi
        self.tdvpEquation = tdvpEquation
        self.stepper = stepper
        self.lindbladian = lindbaldian
        self.measurer = measurer
        if sampler is not None and povm is not None:
            self.conv_measurer = Measurement(sampler, povm)
        else:
            self.conv_measurer = None
        self.max_conv_steps = 10

        self.iteration_count = 0
        self.times = jnp.array([0.])
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

    def initialize(
            self,
            measurestep: int = 0,
            steps: int = 300,
            convergence: bool = False,
            atol: float = 1E-5,
            conv_obs: str = "Sz_i"
    ) -> None:
        """
        Calculates time evolution for a given Lindbladian to obtain its steady state after a set number of `steps`.

        This method also performs measurements during every time step. The wanted observables must be specified
        beforehand in the Measurement object `measurer`. Measurement results will be stored in the object variable
        `results`.

        :param measurestep: Number of steps between measurements. If negative no measurements are performed at all.
        :param steps: Number of time steps.
        :param convergence: Convergence mode time evolves state until specified observable no longer changes. In
        convergence mode the steps parameter will be ignored. Be careful, this does not mean that the state converged
        to the correct steady state, only that it converged to some state.
        :param atol: Absolute tolerance for convergence mode.
        :param conv_obs: Observable that will be checked for convergence. Default is "Sz_i".

        :raises: ValueError
        """
        if convergence:
            if self.conv_measurer is None:
                raise ValueError(f"No POVM or no sampler defined!")
            self.conv_measurer.set_observables([conv_obs])
            if measurestep >= 0:
                if self.measurer is None:
                    raise ValueError(f"Trying to measure every {measurestep} steps while no measurer has been defined "
                                     f"for this initializer.")
                self.__with_measurement_with_conv(measurestep=measurestep, atol=atol, conv_obs=conv_obs)
            else:
                self.__no_measurement_with_conv(atol=atol, conv_obs=conv_obs)
        else:
            if measurestep >= 0:
                if self.measurer is None:
                    raise ValueError(f"Trying to measure every {measurestep} steps while no measurer has been defined "
                                     f"for this initializer.")
                self.__with_measurement_no_conv(measurestep=measurestep, steps=steps)
            else:
                self.__no_measurements_no_conv(steps=steps)

    def __no_measurement_with_conv(self, atol: float, conv_obs: str) -> None:
        prev_res = self.conv_measurer.measure()[conv_obs]

        conv_steps = 0
        while True:
            dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=self.lindbladian,
                                       psi=self.psi)

            self.psi.set_parameters(dp)

            if conv_steps == self.max_conv_steps:
                curr_res = self.conv_measurer.measure()[conv_obs]
                diff = jnp.sqrt(jnp.mean(jnp.abs(curr_res - prev_res)**2))
                if diff < atol:
                    break
                prev_res = curr_res
                conv_steps = 0
            else:
                conv_steps += 1

    def __with_measurement_with_conv(self, measurestep: int, atol: float, conv_obs: str) -> None:
        raise NotImplementedError()

    def __no_measurements_no_conv(self, steps: int) -> None:
        """
        Helper function for initialisation without any measurements. Not intended to be called directly.
        """
        for _ in tqdm(range(steps)):
            dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=self.lindbladian,
                                       psi=self.psi)

            self.psi.set_parameters(dp)

    def __with_measurement_no_conv(self, measurestep: int, steps: int) -> None:
        """
        Helper function for initialisation with measurements. Not intended to be called directly.
        """
        results = {}
        times = []

        # Do measurement on the first state
        _res = self.measurer.measure()
        for obs in self.measurer.observables:
            results[obs] = [_res[obs]]
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
