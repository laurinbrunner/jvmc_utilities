import jax.numpy as jnp
from tqdm import tqdm
from typing import Optional, Union, List, Dict
import jVMC
from jvmc_utilities.measurement import Measurement
import warnings
import time
from clu import metric_writers
import h5py

import jvmc_utilities
import jax


class ConvergenceWarning(Warning):
    """
    Warning for encountering nan-values in the network parameters at time evolution.
    """
    pass


class Initializer:
    def __init__(
            self,
            psi: jVMC.vqs.NQS,
            tdvpEquation: jVMC.util.TDVP,
            stepper: Union[jVMC.util.Euler, jVMC.util.AdaptiveHeun],
            lindbladian: jVMC.operator.POVMOperator,
            measurer: Optional[Measurement] = None,
            sampler: Optional[Union[jVMC.sampler.MCSampler, jVMC.sampler.ExactSampler]] = None,
            povm: Optional[jVMC.operator.POVM] = None
    ) -> None:
        self.psi = psi
        self.tdvpEquation = tdvpEquation
        self.stepper = stepper
        self.lindbladian = lindbladian
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
        warnings.warn("initialize_no_measurement method is deprecated. Using initialize(measure_step=-1) instead is "
                      "adviced.", DeprecationWarning)
        self.initialize(measure_step=-1, steps=steps)

    def initialize(
            self,
            measure_step: int = 0,
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

        :param measure_step: Number of steps between measurements. If negative no measurements are performed at all.
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
            if measure_step >= 0:
                if self.measurer is None:
                    raise ValueError(f"Trying to measure every {measure_step} steps while no measurer has been defined "
                                     f"for this initializer.")
                self.__with_measurement_with_conv(measure_step=measure_step, atol=atol, conv_obs=conv_obs)
            else:
                self.__no_measurement_with_conv(atol=atol, conv_obs=conv_obs)
        else:
            if measure_step >= 0:
                if self.measurer is None:
                    raise ValueError(f"Trying to measure every {measure_step} steps while no measurer has been defined "
                                     f"for this initializer.")
                self.__with_measurement_no_conv(measure_step=measure_step, steps=steps)
            else:
                self.__no_measurements_no_conv(steps=steps)

    def __no_measurement_with_conv(self, atol: float, conv_obs: str) -> None:
        prev_res = self.conv_measurer.measure()[conv_obs]

        conv_steps = 0
        while True:
            dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=self.lindbladian,
                                       psi=self.psi)

            if jnp.any(jnp.isnan(dp)):
                warnings.warn("Initializer ran into nan parameters. Cancelled initialisation.", ConvergenceWarning)
                break

            self.psi.set_parameters(dp)

            if conv_steps == self.max_conv_steps:
                curr_res = self.conv_measurer.measure()[conv_obs]
                diff = self.__convergence_function(prev_res, curr_res)
                if diff < atol:
                    break
                prev_res = curr_res
                conv_steps = 0
            else:
                conv_steps += 1

    def __with_measurement_with_conv(self, measure_step: int, atol: float, conv_obs: str) -> None:
        prev_res = self.conv_measurer.measure()[conv_obs]

        results = {obs: [] for obs in self.measurer.observables}
        times = []

        # Do measurement on the first state
        self.__do_measurement(results, times, self.times[-1])

        t = times[-1]
        try:
            # This try block ensures that the results are saved to the object variables even when the initialisation
            # is cancelled early, either from outside or through a convergence problem
            measure_counter = 0
            conv_steps = 0
            while True:
                dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(),
                                           hamiltonian=self.lindbladian, psi=self.psi)

                if jnp.any(jnp.isnan(dp)):
                    warnings.warn("Initializer ran into nan parameters. Cancelled initialisation.", ConvergenceWarning)
                    break

                t += dt
                self.psi.set_parameters(dp)

                if measure_counter == measure_step:
                    self.__do_measurement(results, times, t)
                    measure_counter = 0
                else:
                    measure_counter += 1

                if conv_steps == self.max_conv_steps:
                    curr_res = self.conv_measurer.measure()[conv_obs]
                    diff = self.__convergence_function(prev_res, curr_res)
                    if diff < atol:
                        break
                    prev_res = curr_res
                    conv_steps = 0
                else:
                    conv_steps += 1
        finally:
            # Make sure that measurement is done on the converged state
            if measure_counter != 0:
                self.__do_measurement(results, times, t)

            self.__convert_to_arrays(results, times)

    def __no_measurements_no_conv(self, steps: int) -> None:
        """
        Helper function for initialisation without any measurements. Not intended to be called directly.
        """
        for _ in tqdm(range(steps)):
            dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=self.lindbladian,
                                       psi=self.psi)

            if jnp.any(jnp.isnan(dp)):
                warnings.warn("Initializer ran into nan parameters. Cancelled initialisation.", ConvergenceWarning)
                break

            self.psi.set_parameters(dp)

    def __with_measurement_no_conv(self, measure_step: int, steps: int) -> None:
        """
        Helper function for initialisation with measurements. Not intended to be called directly.
        """
        results = {obs: [] for obs in self.measurer.observables}
        times = []

        # Do measurement on the first state
        self.__do_measurement(results, times, self.times[-1])

        t = times[-1]

        try:
            # This try block ensures that the results are saved to the object variables even when the initialisation
            # is cancelled early, either from outside or through a convergence problem
            measure_counter = 0
            for _ in tqdm(range(steps)):
                dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(),
                                           hamiltonian=self.lindbladian, psi=self.psi)

                if jnp.any(jnp.isnan(dp)):
                    warnings.warn("Initializer ran into nan parameters. Cancelled initialisation.", ConvergenceWarning)
                    break

                t += dt
                self.psi.set_parameters(dp)

                if measure_counter == measure_step:
                    self.__do_measurement(results, times, t)
                    measure_counter = 0
                else:
                    measure_counter += 1
        finally:
            self.__convert_to_arrays(results, times)

    @staticmethod
    def __convergence_function(a: jnp.ndarray, b: jnp.ndarray) -> float:
        return jnp.sqrt(jnp.mean(jnp.abs(a - b)**2))

    def __do_measurement(self, results: Dict[str, List[jnp.ndarray]], times: List[float], t: float) -> None:
        _res = self.measurer.measure()
        for obs in self.measurer.observables:
            results[obs].append(_res[obs])
        times.append(t)

    def __convert_to_arrays(self, results: Dict[str, List[jnp.ndarray]], times: List[float]) -> None:
        """
        Converts results dictionary and times list to jnp.ndarray and sets them to the respective instance variables.
        """
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


class TimeEvolver:
    """
    Class for evolving a state in time according to a specified Lindblad operator.
    """

    def __init__(
            self,
            psi: jVMC.vqs.NQS,
            tdvpEquation: jVMC.util.TDVP,
            stepper: Union[jVMC.util.Euler, jVMC.util.AdaptiveHeun],
            measurer: Measurement,
            writer: metric_writers.summary_writer.SummaryWriter = None,
            additional_hparams: Dict = None,
            parameter_file: str = None
    ) -> None:
        self.psi = psi
        self.tdvpEquation = tdvpEquation
        self.stepper = stepper
        if type(stepper) == jVMC.util.Euler:
            self.adaptive_stepper = False
        else:
            self.adaptive_stepper = True

        self.measurer = measurer
        self.writer = writer
        self.additional_hparams = additional_hparams
        self.write_index = 0

        if parameter_file is not None:
            try:
                _f = h5py.File(parameter_file, 'r')
                runs = list(_f.keys())
                for i, r in enumerate(runs):
                    runs[i] = int(r.split("_")[-1])
                current_run = max(runs) + 1
                group = f"run_{current_run}"
                _f.close()
            except FileNotFoundError:
                group = "run_1"

            if self.additional_hparams is not None:
                self.additional_hparams["parameter_output_run"] = group
            else:
                self.additional_hparams = {"parameter_output_run": group}

            self.parameter_output_manager = jVMC.util.OutputManager(parameter_file, append=True, group=group)
        else:
            self.parameter_output_manager = None

        if writer is not None:
            self.__write_hparams()

        self.real_times = []  # Every list in this list is for potential reruns
        self.times = jnp.array([0.])
        self.integrated_tdvpError = 0.
        self.integrated_cvError = 0.
        self.results = {}
        self.meta_data = {"tdvp_errors": None, "tdvp_residuals": None, "cv_errors": None, "cv_residuals": None}

    def run(self, lindbladian: jVMC.operator.POVMOperator, max_time: float, measure_step: int = 0) -> None:

        results = {obs: [] for obs in self.measurer.observables}
        times = []
        self.real_times.append([])
        meta_data = {"tdvp_errors": [], "tdvp_residuals": [], "cv_errors": [], "cv_residuals": []}

        self.__do_measurement(results, times, self.times[-1], meta_data)

        t = times[0]

        pbar = tqdm(total=100, desc="Progress", unit="%")
        bar_index = 1
        measure_counter = 0
        try:
            while t - times[0] < max_time:

                if self.adaptive_stepper:
                    dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(),
                                               hamiltonian=lindbladian, psi=self.psi, normFunction=self.__norm_fun)
                else:
                    dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(),
                                               hamiltonian=lindbladian, psi=self.psi)

                if jnp.any(jnp.isnan(dp)):
                    warnings.warn("TimeEvolver ran into nan-valued parameters. Aborted time evolution.",
                                  ConvergenceWarning)
                    break

                t += dt
                self.psi.set_parameters(dp)

                if measure_counter == measure_step:
                    self.__do_measurement(results, times, t, meta_data)
                    measure_counter = 0
                else:
                    measure_counter += 1

                # update tqdm bar
                if t - times[0] > max_time / 100 * bar_index:
                    old_bar_index = bar_index
                    bar_index = int(jnp.floor(t / max_time * 100) + 1)
                    pbar.update(bar_index - old_bar_index)
                    pbar.set_postfix({"t": t})

        finally:
            pbar.close()

            # Make sure that measurement is done at the last step
            if measure_counter != 0:
                self.__do_measurement(results, times, t, meta_data)

            self.__convert_to_arrays(results, times, meta_data)

    def __norm_fun(self, v: jnp.ndarray) -> float:
        return jnp.real(jnp.conj(jnp.transpose(v)).dot(self.tdvpEquation.S_dot(v)))

    def __write(
            self,
            results: Dict[str, List[jnp.ndarray]],
            t: float, dt: float,
            tdvp_errs: List[float],
            cv_errs: List[float]
    ) -> None:
        writedict = {}
        if "N" in results.keys():
            writedict["N"] = results["N"][0] + results["N"][1]
            writedict["M"] = results["N"][0] - results["N"][1]
            if self.measurer.mc_errors:
                writedict["N_MC_error"] = results["N_MC_error"][0] + results["N_MC_error"][1]
                writedict["M_MC_error"] = results["N_MC_error"][0] - results["N_MC_error"][1]
        if "Sx_i" in results.keys():
            writedict["X"] = jnp.mean(results["Sx_i"])
            for i in range(results["Sx_i"].shape[0]):
                writedict[f"Sx_i/{i}"] = results["Sx_i"][i]
            if self.measurer.mc_errors:
                writedict["X_MC_error"] = jnp.mean(results["Sx_i_MC_error"])
                for i in range(results["Sx_i_MC_error"].shape[0]):
                    writedict[f"Sx_i_MC_error/{i}"] = results["Sx_i_MC_error"][i]
        if "Sy_i" in results.keys():
            writedict["Y"] = jnp.mean(results["Sy_i"])
            for i in range(results["Sy_i"].shape[0]):
                writedict[f"Sy_i/{i}"] = results["Sy_i"][i]
            if self.measurer.mc_errors:
                writedict["Y_MC_error"] = jnp.mean(results["Sy_i_MC_error"])
                for i in range(results["Sy_i"].shape[0]):
                    writedict[f"Sy_i_MC_error/{i}"] = results["Sy_i_MC_error"][i]
        if "Sz_i" in results.keys():
            writedict["Z"] = jnp.mean(results["Sz_i"])
            for i in range(results["Sz_i"].shape[0]):
                writedict[f"Sz_i/{i}"] = results["Sz_i"][i]
            if self.measurer.mc_errors:
                writedict["Z_MC_error"] = jnp.mean(results["Sz_i_MC_error"])
                for i in range(results["Sz_i"].shape[0]):
                    writedict[f"Sz_i_MC_error/{i}"] = results["Sz_i_MC_error"][i]
        if "M_sq" in results.keys():
            writedict["M_sq"] = results["M_sq"]
        if "m_corr" in results.keys():
            L = results["m_corr"].shape[0]
            for i in range(L):
                for j in range(L):
                    writedict[f"m_corr/{i},{j}"] = results["m_corr"][i, j]
        if "N_i" in results.keys():
            system_L = results["N_i"].shape[0] // 2
            for l in range(system_L):
                writedict[f"N_l/{l}_up"] = results["N_i"][2*l]
                writedict[f"N_l/{l}_down"] = results["N_i"][2*l+1]
                writedict[f"M_l/{l}"] = results["N_i"][2*l] - results["N_i"][2*l+1]
                if self.measurer.mc_errors:
                    writedict[f"N_l_MC_error/{l}_up"] = results["N_i_MC_error"][2*l]
                    writedict[f"N_l_MC_error/{l}_down"] = results["N_i_MC_error"][2*l+1]
                    writedict[f"M_l_MC_error/{l}"] = results["N_i_MC_error"][2*l] - results["N_i_MC_error"][2*l+1]

        self.writer.write_scalars(self.write_index, {"dt": dt, "t": t, "tdvp_Error": tdvp_errs[0],
                                                     "tdvp_Residual": tdvp_errs[1], "CV_Error": cv_errs[0],
                                                     "tdvp_Error/integrated": self.integrated_tdvpError,
                                                     "CV_Error/integrated": self.integrated_cvError,
                                                     "CV_Residual": cv_errs[1]})
        try:
            snr = self.tdvpEquation.get_snr()
            spectrum = self.tdvpEquation.get_spectrum()

            self.writer.write_scalars(self.write_index, {"SNR/mean": jnp.mean(snr),
                                                         "SNR/logmean": jnp.mean(jnp.log(snr))})

            writedict["SNR/mean_time"] = jnp.mean(snr)
            writedict["SNR/logmean_time"] = jnp.mean(jnp.log(snr))

            self.writer.write_histograms(self.write_index, {"SNR": snr, "Spectrum": spectrum,
                                                            "logSNR": jnp.log10(jnp.abs(snr) + 1E-18),
                                                            "logSpectrum": jnp.log10(jnp.abs(spectrum) + 1E-18)})
            self.writer.write_histograms(jnp.floor(1E6*t), {"SNR/time": snr, "Spectrum/time": spectrum,
                                                            "logSNR/time": jnp.log10(jnp.abs(snr) + 1E-18),
                                                            "logSpectrum/time": jnp.log10(jnp.abs(spectrum) + 1E-18)})
        except TypeError:
            pass

        writedict["tdvp_Error/time"] = tdvp_errs[0]
        writedict["tdvp_Residual/time"] = tdvp_errs[1]
        writedict["tdvp_Error/integrated_time"] = self.integrated_tdvpError
        writedict["CV_Error/time"] = cv_errs[0]
        writedict["CV_Residual/time"] = cv_errs[1]
        writedict["CV_Error/integrated_time"] = self.integrated_cvError
        writedict["dt/time"] = dt

        self.write_index += 1
        self.writer.write_scalars(jnp.floor(1E6*t), writedict)

    def __do_measurement(
            self,
            results: Dict[str, List[jnp.ndarray]],
            times: List[float],
            t: float,
            meta_data: Dict[str, List[float]]
    ) -> None:
        self.real_times[-1].append(time.time())
        _res = self.measurer.measure()
        for obs in self.measurer.observables:
            results[obs].append(_res[obs])
        times.append(t)

        if len(times) == 1:
            dt = times[-1]
        else:
            dt = times[-1] - times[-2]

        if self.tdvpEquation.metaData is not None:
            # This will be skipped when the TDVP.__call__ function has not been called yet
            td_errs = self.tdvpEquation.get_residuals()
            if self.tdvpEquation.crossValidation:
                cv_errs = [self.tdvpEquation.crossValidationFactor_tdvpErr,
                           self.tdvpEquation.crossValidationFactor_residual]
            else:
                cv_errs = [0., 0.]
        else:
            td_errs = [0., 0.]
            cv_errs = [0., 0.]

        self.integrated_tdvpError += dt * td_errs[0]
        self.integrated_cvError += dt * cv_errs[0]

        meta_data["tdvp_errors"].append(td_errs[0])
        meta_data["tdvp_residuals"].append(td_errs[1])
        meta_data["cv_errors"].append(cv_errs[0])
        meta_data["cv_residuals"].append(cv_errs[1])

        if self.writer is not None:
            self.__write(_res, t, dt, td_errs, cv_errs)
        if self.parameter_output_manager is not None:
            self.__save_parameters(t)

    def __convert_to_arrays(
            self,
            results: Dict[str, List[jnp.ndarray]],
            times: List[float],
            meta_data: Dict[str, List[float]]
    ) -> None:
        """
        Converts results dictionary and times list to jnp.ndarray and sets them to the respective instance variables.
        """
        if len(self.results.keys()) == 0:
            self.results = {}
            for obs in results.keys():
                self.results[obs] = jnp.array(results[obs])
            self.times = jnp.array(times)
            self.meta_data["tdvp_errors"] = jnp.array(meta_data["tdvp_errors"])
            self.meta_data["tdvp_residuals"] = jnp.array(meta_data["tdvp_residuals"])
            self.meta_data["cv_errors"] = jnp.array(meta_data["cv_errors"])
            self.meta_data["cv_residuals"] = jnp.array(meta_data["cv_residuals"])
        else:
            for obs in results.keys():
                if obs in self.results.keys():
                    self.results[obs] = jnp.concatenate([self.results[obs], jnp.array(results[obs])])
                else:
                    self.results[obs] = jnp.array(results[obs])
            self.times = jnp.concatenate([self.times, jnp.array(times)])
            self.meta_data["tdvp_errors"] = jnp.concatenate([self.meta_data["tdvp_errors"],
                                                             jnp.array(meta_data["tdvp_errors"])])
            self.meta_data["tdvp_residuals"] = jnp.concatenate([self.meta_data["tdvp_residuals"],
                                                                jnp.array(meta_data["tdvp_residuals"])])
            self.meta_data["cv_errors"] = jnp.concatenate([self.meta_data["cv_errors"],
                                                           jnp.array(meta_data["cv_errors"])])
            self.meta_data["cv_residuals"] = jnp.concatenate([self.meta_data["cv_residuals"],
                                                              jnp.array(meta_data["cv_residuals"])])
        self.real_times[-1] = jnp.array(self.real_times[-1])
        self.real_times[-1] = self.real_times[-1] - self.real_times[-1][0]

    def __save_parameters(self, t: float) -> None:
        """
        Save network parameters of neural quantum state.
        """
        self.parameter_output_manager.write_network_checkpoint(t, self.psi.get_parameters())

    def __write_hparams(self) -> None:
        """
        Write hyperparameters to tensorboard file.
        """
        hparams = {"system_size": self.tdvpEquation.sampler.sampleShape[0],
                   "network": str(type(self.psi.net)),
                   "seed": self.psi.seed,
                   "sampler": str(type(self.tdvpEquation.sampler)),
                   "stepper": str(type(self.stepper)),
                   "snrTol": self.tdvpEquation.snrTol,
                   "svdTol": self.tdvpEquation.svdTol,
                   "batchSize": self.psi.batchSize,
                   "jVMC_version": jVMC.__version__,
                   "jvmc_utilities_version": jvmc_utilities.__version__}

        net_params = vars(self.psi.net)
        for k in net_params:
            if k in ["name", "parent", "_state", "_id"]:
                continue
            elif k == "actFun":
                hparams[k] = net_params[k].__name__
            elif k == "orbit":
                if net_params[k] is None:
                    hparams[k] = "not symmetric"
                else:
                    hparams[k] = "symmetric"
            else:
                hparams[k] = net_params[k]

        if type(self.tdvpEquation.sampler) is jVMC.sampler.MCSampler:
            hparams["sample_size"] = self.tdvpEquation.sampler.numSamples

        if type(self.stepper) is jVMC.util.stepper.AdaptiveHeun:
            hparams["integration_tolerance"] = self.stepper.tolerance

        if self.additional_hparams is not None:
            for k in self.additional_hparams.keys():
                hparams[k] = self.additional_hparams[k]

        self.writer.write_hparams(hparams)


def copy_state(source: jVMC.vqs.NQS, target: jVMC.vqs.NQS) -> None:
    target.set_parameters(source.get_parameters())


if __name__ == '__main__':
    L = 4
    prngkey = jax.random.PRNGKey(0)
    cnn = jvmc_utilities.nets.POVMCNN(L=L)  # , depth=3, features=(8, 8))
    psi = jVMC.vqs.NQS(cnn, seed=1234)
    sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)
    # sampler = jVMC.sampler.MCSampler(psi, (L,), prngkey, numSamples=2000)
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1.,
                                       svdTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=True)
    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)
    # stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=1e-3, tol=1E-6)
    povm = jVMC.operator.POVM({"dim": "1D", "L": L})
    lind = jVMC.operator.POVMOperator(povm)
    jvmc_utilities.operators.initialisation_operators(povm)
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (0, 1)})
    lind.add({"name": "updown_dis", "strength": 5.0, "sites": (2, 3)})
    measurer = jvmc_utilities.measurement.Measurement(sampler, povm)
    measurer.set_observables(["Sz_i"])

    evol = TimeEvolver(psi, tdvpEquation, stepper, measurer, writer=None, parameter_file="test")
    evol.run(lind, 1)
