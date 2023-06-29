import jax.numpy as jnp
from tqdm import tqdm
from typing import Optional, Union, List, Dict
import jVMC
from jvmc_utilities.measurement import Measurement
import warnings
import time
from clu import metric_writers
import h5py
import os

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
            parameter_file: str = None,
            timing_file: str = None
    ) -> None:
        self.psi = psi
        self.tdvpEquation = tdvpEquation
        self.stepper = stepper

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
        self.results = {}
        self.meta_data = {"tdvp_Error": None, "tdvp_Residual": None, "CV_Error": None, "CV_Residual": None,
                          "tdvp_Error/integrated": None, "CV_Error/integrated": None, "ElocMean": None, "ElocVar": None}

        if timing_file is not None:
            # Create timing output manager, that can be deleted right away (only the object is needed, not the file)
            self.timing_manager = jVMC.util.OutputManager("./timing.tmp")
            try:
                os.remove("./timing.tmp")
            except OSError:
                pass

            self.timings = {}

        self.timing_file = timing_file

        # Dictionaries for current run
        self.__reset_current_run_dicts()

    def __reset_current_run_dicts(self) -> None:
        self.current_meta_data = {"tdvp_Error": [], "tdvp_Residual": [], "CV_Error": [], "CV_Residual": [],
                                  "ElocMean": [], "ElocVar": [], "tdvp_Error/integrated": [], "CV_Error/integrated": []}
        self.current_results = {obs: [] for obs in self.measurer.observables}
        self.current_times = []



    def run(self, lindbladian: jVMC.operator.POVMOperator, max_time: float, measure_step: int = 0) -> None:

        def start_timing(name: str) -> None:
            if self.timing_file is not None:
                self.timing_manager.start_timing(name)

        def stop_timing(name: str) -> None:
            if self.timing_file is not None:
                self.timing_manager.stop_timing(name)

        self.real_times.append([])

        self.__do_measurement(t=0., dt=0.)

        t = self.current_times[0]

        pbar = tqdm(total=100, desc="Progress", unit="%")
        bar_index = 1
        measure_counter = 0
        try:
            while t - self.current_times[0] < max_time:

                dp, dt = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(),
                                           hamiltonian=lindbladian, psi=self.psi, normFunction=self.__norm_fun,
                                           outp=self.timing_manager if self.timing_file is not None else None)

                if jnp.any(jnp.isnan(dp)):
                    warnings.warn("TimeEvolver ran into nan-valued parameters. Aborted time evolution.",
                                  ConvergenceWarning)
                    break

                t += dt
                self.psi.set_parameters(dp)

                start_timing("TimeEvolver measurement")
                if measure_counter == measure_step:
                    self.__do_measurement(t=t, dt=dt)
                    measure_counter = 0
                else:
                    measure_counter += 1
                stop_timing("TimeEvolver measurement")

                # Save parameters at every step
                if self.parameter_output_manager is not None:
                    self.__save_parameters(t)
                if self.timing_file is not None:
                    self.__take_timings()

                # update tqdm bar
                pbar.set_postfix({"t": t})
                if t - self.current_times[0] > max_time / 100 * bar_index:
                    old_bar_index = bar_index
                    bar_index = int(jnp.floor(t / max_time * 100) + 1)
                    pbar.update(bar_index - old_bar_index)
                    pbar.set_postfix({"t": t})

        finally:
            pbar.close()

            # Make sure that measurement is done at the last step
            if measure_counter != 0:
                self.__do_measurement(t=t, dt=dt)

            self.__convert_to_arrays()
            if self.timing_file is not None:
                self.__save_timings()

    def __norm_fun(self, v: jnp.ndarray) -> float:
        return jnp.real(jnp.conj(jnp.transpose(v)).dot(self.tdvpEquation.S_dot(v)))

    def __write(
            self,
            results: Dict[str, List[jnp.ndarray]],
            t: float, dt: float,
            meta_data_update: Dict[str, float]
    ) -> None:
        writedict_time = {}
        writedict_index = {}

        # Write standard observables
        for obs in ["Sx_i", "Sy_i", "Sz_i", "m_corr", "Sx_i_MC_error", "Sy_i_MC_error", "Sz_i_MC_error", "M_sq"]:
            try:
                if len(results[obs].shape) == 1:
                    writedict_time[obs[1].upper() + obs[4:]] = jnp.mean(results[obs])
                    for i in range(results[obs].shape[0]):
                        writedict_time[f"{obs}/{i}"] = results[obs][i]
                elif len(results[obs].shape) == 2:
                    for i in range(results[obs].shape[0]):
                        for j in range(results[obs].shape[1]):
                            writedict_time[f"{obs}/{i},{j}"] = results[obs][i, j]
                else:
                    writedict_time[f"{obs}"] = results[obs]
            except KeyError:
                pass

        # Write occupation observables
        if "N" in results.keys():
            writedict_time["N"] = results["N"][0] + results["N"][1]
            writedict_time["M"] = results["N"][0] - results["N"][1]
            if self.measurer.mc_errors:
                writedict_time["N_MC_error"] = results["N_MC_error"][0] + results["N_MC_error"][1]
                writedict_time["M_MC_error"] = results["N_MC_error"][0] - results["N_MC_error"][1]
        if "N_i" in results.keys():
            system_L = results["N_i"].shape[0] // 2
            for l in range(system_L):
                writedict_time[f"N_l/{l}_up"] = results["N_i"][2*l]
                writedict_time[f"N_l/{l}_down"] = results["N_i"][2*l+1]
                writedict_time[f"M_l/{l}"] = results["N_i"][2*l] - results["N_i"][2*l+1]
                if self.measurer.mc_errors:
                    writedict_time[f"N_l_MC_error/{l}_up"] = results["N_i_MC_error"][2*l]
                    writedict_time[f"N_l_MC_error/{l}_down"] = results["N_i_MC_error"][2*l+1]
                    writedict_time[f"M_l_MC_error/{l}"] = results["N_i_MC_error"][2*l] - results["N_i_MC_error"][2*l+1]

        # Write meta data
        writedict_index["t"] = t
        writedict_index["dt"] = dt
        writedict_time["dt/time"] = dt
        for key in meta_data_update.keys():
            writedict_index[key] = meta_data_update[key]
            if "/" in key:
                writedict_time[f"{key}_time"] = meta_data_update[key]
            else:
                writedict_time[f"{key}/time"] = meta_data_update[key]

        if self.tdvpEquation.metaData is not None:
            # This will be skipped when the TDVP.__call__ function has not been called yet
            snr = self.tdvpEquation.get_snr()
            spectrum = self.tdvpEquation.get_spectrum()

            writedict_index["SNR/mean"] = jnp.mean(snr)
            writedict_index["SNR/logmean"] = jnp.mean(jnp.log(snr))
            writedict_time["SNR/mean_time"] = jnp.mean(snr)
            writedict_time["SNR/logmean_time"] = jnp.mean(jnp.log(snr))

            self.writer.write_histograms(self.write_index, {"SNR": snr, "Spectrum": spectrum,
                                                            "logSNR": jnp.log10(jnp.abs(snr) + 1E-18),
                                                            "logSpectrum": jnp.log10(jnp.abs(spectrum) + 1E-18)})
            self.writer.write_histograms(jnp.floor(1E6*t), {"SNR/time": snr, "Spectrum/time": spectrum,
                                                            "logSNR/time": jnp.log10(jnp.abs(snr) + 1E-18),
                                                            "logSpectrum/time": jnp.log10(jnp.abs(spectrum) + 1E-18)})

        self.write_index += 1
        self.writer.write_scalars(self.write_index, writedict_index)
        self.writer.write_scalars(jnp.floor(1E6*t), writedict_time)

    def __do_measurement(self, t: float, dt: float) -> None:
        """
        Measure specified observables and meta data. Afterward, write them to the tensorboard writer.
        """
        self.real_times[-1].append(time.time())

        # Measure observables
        _res = self.measurer.measure()
        for obs in self.measurer.observables:
            self.current_results[obs].append(_res[obs])

        self.current_times.append(t)

        # Measure meta data
        if self.tdvpEquation.metaData is not None:
            # This will be skipped when the TDVP.__call__ function has not been called yet
            td_errs = self.tdvpEquation.get_residuals()
            ElocMean = self.tdvpEquation.ElocMean0
            ElocVar = self.tdvpEquation.ElocVar0
            if len(self.current_meta_data["tdvp_Error/integrated"]) != 0:
                new_integrated_tdvp_error = self.current_meta_data["tdvp_Error/integrated"][-1] + dt * td_errs[0]
                new_integrated_cv_error = self.current_meta_data["CV_Error/integrated"][-1] + dt * td_errs[0]
            else:
                new_integrated_tdvp_error = dt * td_errs[0]
                new_integrated_cv_error = dt * td_errs[0]

            if self.tdvpEquation.crossValidation:
                cv_errs = [self.tdvpEquation.crossValidationFactor_tdvpErr,
                           self.tdvpEquation.crossValidationFactor_residual]
            else:
                cv_errs = [0., 0.]
        else:
            ElocMean = 0.
            ElocVar = 0.
            td_errs = [0., 0.]
            cv_errs = [0., 0.]
            new_integrated_tdvp_error = 0.
            new_integrated_cv_error = 0.

        meta_data_update = {"tdvp_Error/integrated": new_integrated_tdvp_error,
                            "CV_Error/integrated": new_integrated_cv_error,
                            "tdvp_Error": td_errs[0],
                            "tdvp_Residual": td_errs[1],
                            "CV_Error": cv_errs[0],
                            "CV_Residual": cv_errs[1],
                            "ElocMean": ElocMean,
                            "ElocVar": ElocVar}

        for key in meta_data_update.keys():
            self.current_meta_data[key].append(meta_data_update[key])

        if self.writer is not None:
            self.__write(_res, t, dt, meta_data_update)

    def __convert_to_arrays(self) -> None:
        """
        Converts results dictionary and times list to jnp.ndarray and sets them to the respective instance variables.
        """
        if len(self.results.keys()) == 0:
            self.results = {}
            for obs in self.current_results.keys():
                self.results[obs] = jnp.array(self.current_results[obs])
            self.times = jnp.array(self.current_times)
            for key in self.current_meta_data.keys():
                self.meta_data[key] = jnp.array(self.current_meta_data[key])
        else:
            for obs in self.current_results.keys():
                if obs in self.results.keys():
                    self.results[obs] = jnp.concatenate([self.results[obs], jnp.array(self.current_results[obs])])
                else:
                    self.results[obs] = jnp.array(self.current_results[obs])
            self.times = jnp.concatenate([self.times, jnp.array(self.current_times)])
            for key in self.current_meta_data.keys():
                self.meta_data[key] = jnp.concatenate([self.meta_data[key], jnp.array(self.current_meta_data[key])])
        self.real_times[-1] = jnp.array(self.real_times[-1])
        self.real_times[-1] = self.real_times[-1] - self.real_times[-1][0]

        # Reset current run dictionaries
        self.__reset_current_run_dicts()

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
                   "pinvTol": self.tdvpEquation.pinvTol,
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

    def __take_timings(self) -> None:
        if not self.timings:
            for key, item in self.timing_manager.timings.items():
                self.timings[key] = {"count": [], "total": [], "time": []}
            self.timings["count"] = 1
        for key, item in self.timing_manager.timings.items():
            for k, k2 in [("count", "count"), ("total", "total"), ("time", "newest")]:
                self.timings[key][k].append(item[k2])
        self.timings["count"] += 1

    def __save_timings(self) -> None:
        with open(self.timing_file, "w" ) as f:
            f.write(f"{'' :<40}{'Total' :<25}{'per step' :<25}\n")
            f.write("-" * 90 + "\n")
            for i in range(self.timings["count"] - 1):
                timing_string = f"count: {i}\n"
                for key in self.timings.keys() - ["count"]:
                    timing_string += f"{key :<40}{self.timings[key]['total'][i] :<25}" \
                                     f"{self.timings[key]['time'][i] :<25}\n"
                timing_string += "-" * 90 + "\n"
                f.write(timing_string)


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
