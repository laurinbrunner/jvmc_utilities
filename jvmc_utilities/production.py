import jax.numpy as jnp
import h5py
import jVMC
import jVMC.mpi_wrapper as mpi
from typing import Optional, Union, List, Dict, Callable, Tuple
import jvmc_utilities
from jvmc_utilities import nets as ju_nets
from flax import linen as nn
import jax
import argparse


def lindblad_initialisation(
        povm: jVMC.operator.POVM,
        eps_0: float,
        eps_1: float,
        eps_2: float,
        system_size: int,
        ElocBatchSize: int = -1
) -> jVMC.operator.POVMOperator:
    lindblad = jVMC.operator.POVMOperator(povm, ElocBatchSize=ElocBatchSize)

    strength = 4.8  # arbitrary variable that controlls the speed of thermalisation, changing it helps in case of convergence problems

    for l in range(system_size):
        lindblad.add({"name": "upup_dis", "strength": strength * eps_0 / 2, "sites": (2 * l, 2 * l + 1)})
        lindblad.add({"name": "updown_dis", "strength": strength * eps_1, "sites": (2 * l, 2 * l + 1)})
        lindblad.add({"name": "downup_dis", "strength": strength * eps_2, "sites": (2 * l, 2 * l + 1)})
        lindblad.add({"name": "downdown_dis", "strength": strength * (1 - eps_1 - eps_2 - eps_0 / 2),
                      "sites": (2 * l, 2 * l + 1)})
    return lindblad


def _lindblad_time_evolution_fill_out(
        operator: Union[jVMC.operator.POVMOperator, jvmc_utilities.operators.EfficientPOVMOperator],
        h_transverse: float,
        gamma: float,
        system_size: int
) -> Union[jVMC.operator.POVMOperator, jvmc_utilities.operators.EfficientPOVMOperator]:
    for l in range(system_size):
        # Transverse field H = -h_transverse \sum_l c^+_{l\uparrow}c^-_{l\downarrow} + c^+_{l\downarrow}c^-_{l\uparrow}
        operator.add({"name": "spin_flip_uni", "strength": -h_transverse, "sites": (2 * l, 2 * l + 1)})
        operator.add({"name": "inv_spin_flip_uni", "strength": -h_transverse, "sites": (2 * l, 2 * l + 1)})

        # restricted directed hopping L_{l\uparrow} = c^+_{l\uparrow}(1 - n_{l\downarrow})c^-_{l+1\uparrow}
        operator.add({"name": "restricted_jump_dis", "strength": gamma,
                      "sites": (2 * l, (2 * l + 1) % (2 * system_size), (2 * l + 2) % (2 * system_size))})

        # restricted directed hopping L_{l\downarrow} = c^+_{l+1\downarrow}(1 - n_{l+1\uparrow})c^-_{l\downarrow}
        operator.add({"name": "restricted_jump_dis", "strength": gamma,
                      "sites": ((2 * l + 3) % (2 * system_size), (2 * l + 2) % (2 * system_size),
                                (2 * l + 1) % (2 * system_size))})
    return operator


def lindblad_time_evolution_efficient(
        povm: jVMC.operator.POVM,
        h_transverse: float,
        gamma: float,
        system_size: int,
        ElocBatchSize: int = -1
) -> jvmc_utilities.operators.EfficientPOVMOperator:
    Lindbladian = jvmc_utilities.operators.EfficientPOVMOperator(povm=povm, ElocBatchSize=ElocBatchSize)
    return _lindblad_time_evolution_fill_out(operator=Lindbladian,
                                             h_transverse=h_transverse,
                                             gamma=gamma,
                                             system_size=system_size)


def lindblad_time_evolution(
        povm: jVMC.operator.POVM,
        h_transverse: float,
        gamma: float,
        system_size: int,
        ElocBatchSize: int = -1
) -> jVMC.operator.POVMOperator:
    Lindbladian = jVMC.operator.POVMOperator(povm=povm, ElocBatchSize=ElocBatchSize)
    return _lindblad_time_evolution_fill_out(operator=Lindbladian,
                                             h_transverse=h_transverse,
                                             gamma=gamma,
                                             system_size=system_size)


class NetworkHyperparameters:
    """Dataclass for all network hyperparameters"""
    def __init__(
            self,
            network_type: str,
            param_dtype: str,
            depth: int,
            features: int,
            kernel_size: int,
            embeddingDimFac: int,
            attention_heads: int,
            symmetry: bool
    ):
        self.network_type = network_type
        self.param_dtype = param_dtype
        self.depth = depth
        self.features = features
        self.kernel_size = kernel_size
        self.embeddingDimFac = embeddingDimFac
        self.attention_heads = attention_heads
        self.symmetry = symmetry

    def network_string(self) -> str:
        out_string = f"{self.network_type}_d{self.depth}_f{self.features}"
        if self.network_type in ["CNN", "GatedCNN", "CNNResidual", "CNNEmbedded", "CNNEmbeddedResidual", "CNNAttention",
                                 "CNNAttentionResidual", "CNN_mcmc", "ResNet_mcmc"]:
            out_string += f"_ks{self.kernel_size}"
        if self.network_type in ["CNNEmbedded", "CNNEmbeddedResidual"]:
            out_string += f"_embDim{self.embeddingDimFac}"
        if self.network_type in ["CNNAttention", "CNNAttentionResidual"]:
            out_string += f"_attheads{self.attention_heads}"
        if self.symmetry:
            out_string += "_symm"

        out_string += f"_dtype{self.param_dtype}"

        return out_string


def get_network(
        hyperparameters: NetworkHyperparameters,
        system_size: int,
        logProbFactor: float,
        inputDim: int
) -> nn.Module:
    if hyperparameters.param_dtype == "float32":
        dtype = jnp.float32
    elif hyperparameters.param_dtype == "float64":
        dtype = jnp.float64
    else:
        raise ValueError("Unknown dtype")

    if hyperparameters.symmetry:
        orbit = jvmc_utilities.symmetry.get_orbit_aqi_translation(system_size)
    else:
        orbit = None

    if hyperparameters.network_type == "CNN":
        net = ju_nets.POVMCNN(L=2*system_size,
                              depth=hyperparameters.depth,
                              features=hyperparameters.features,
                              orbit=orbit,
                              kernel_size=(hyperparameters.kernel_size,),
                              param_dtype=dtype)
    elif hyperparameters.network_type == "GatedCNN":
        net = ju_nets.POVMCNNGated(L=2*system_size,
                                   depth=hyperparameters.depth,
                                   features=hyperparameters.features,
                                   orbit=orbit,
                                   kernel_size=(hyperparameters.kernel_size,),
                                   param_dtype=dtype)
    elif hyperparameters.network_type == "CNNResidual":
        net = ju_nets.POVMCNNResidual(L=2*system_size,
                                      depth=hyperparameters.depth,
                                      features=hyperparameters.features,
                                      orbit=orbit,
                                      kernel_size=(hyperparameters.kernel_size,),
                                      param_dtype=dtype)
    elif hyperparameters.network_type == "CNNEmbedded":
        net = ju_nets.POVMCNNEmbedded(L=system_size,
                                      depth=hyperparameters.depth,
                                      features=hyperparameters.features,
                                      orbit=orbit,
                                      kernel_size=(hyperparameters.kernel_size,),
                                      param_dtype=dtype,
                                      embeddingDimFac=hyperparameters.embeddingDimFac)
    elif hyperparameters.network_type == "CNNEmbeddedResidual":
        net = ju_nets.POVMCNNEmbeddedResidual(L=system_size,
                                              depth=hyperparameters.depth,
                                              features=hyperparameters.features,
                                              orbit=orbit,
                                              kernel_size=(hyperparameters.kernel_size,),
                                              param_dtype=dtype,
                                              embeddingDimFac=hyperparameters.embeddingDimFac)
    elif hyperparameters.network_type == "CNNAttention":
        net = ju_nets.CNNAttention(L=2*system_size,
                                   depth=hyperparameters.depth,
                                   features=hyperparameters.features,
                                   orbit=orbit,
                                   attention_heads=hyperparameters.attention_heads,
                                   kernel_size=(hyperparameters.kernel_size,),
                                   param_dtype=dtype)
    elif hyperparameters.network_type == "CNNAttentionResidual":
        net = ju_nets.CNNAttentionResidual(L=2*system_size,
                                           depth=hyperparameters.depth,
                                           features=hyperparameters.features,
                                           orbit=orbit,
                                           attention_heads=hyperparameters.attention_heads,
                                           kernel_size=(hyperparameters.kernel_size,),
                                           param_dtype=dtype)
    elif hyperparameters.network_type == "DeepNADE":
        net = ju_nets.DeepNADE(L=2*system_size,
                               depth=hyperparameters.depth,
                               hiddenSize=hyperparameters.features,
                               orbit=orbit)
    elif hyperparameters.network_type == "AFFN":
        net = ju_nets.AFFN(L=2*system_size,
                           depth=hyperparameters.depth,
                           hiddenSize=hyperparameters.features,
                           orbit=orbit)
    elif hyperparameters.network_type == "RNN":
        if hyperparameters.symmetry:
            net = jVMC.nets.RNN1DGeneralSym(L=2*system_size,
                                            depth=hyperparameters.depth,
                                            hiddenSize=hyperparameters.features,
                                            orbit=orbit,
                                            logProbFactor=logProbFactor,
                                            realValuedOutput=True,
                                            inputDim=inputDim,
                                            cell="RNN")
        else:
            net = jVMC.nets.RNN1DGeneral(L=2*system_size,
                                         depth=hyperparameters.depth,
                                         inputDim=inputDim,
                                         hiddenSize=hyperparameters.features,
                                         logProbFactor=logProbFactor,
                                         realValuedOutput=True,
                                         cell="RNN")
    elif hyperparameters.network_type == "LSTM":
        if hyperparameters.symmetry:
            net = jVMC.nets.RNN1DGeneralSym(L=2*system_size,
                                            depth=hyperparameters.depth,
                                            inputDim=inputDim,
                                            hiddenSize=hyperparameters.features,
                                            orbit=orbit,
                                            logProbFactor=logProbFactor,
                                            realValuedOutput=True,
                                            cell="LSTM")
        else:
            net = jVMC.nets.RNN1DGeneral(L=2*system_size,
                                         depth=hyperparameters.depth,
                                         cell="LSTM",
                                         inputDim=inputDim,
                                         hiddenSize=hyperparameters.features,
                                         logProbFactor=logProbFactor,
                                         realValuedOutput=True)
    elif hyperparameters.network_type == "CNN_mcmc":
        net = ju_nets.MCMC_CNN(features=hyperparameters.features,
                               depth=hyperparameters.depth,
                               kernel_size=(hyperparameters.kernel_size,),
                               param_dtype=dtype)
    elif hyperparameters.network_type == "ResNet_mcmc":
        net = ju_nets.MCMC_ResNet(features=hyperparameters.features,
                                  depth=hyperparameters.depth,
                                  kernel_size=(hyperparameters.kernel_size, ),
                                  param_dtype=dtype)
    else:
        raise ValueError("Unknown network type. Valid types are 'GatedCNN', 'CNN', 'DeepNADE', 'AFFN', 'RNN', 'LSTM', "
                         "'CNN_mcmc', 'CNNResidual', 'CNNEmbedded', 'CNNEmbeddedResidual', 'CNNAttention', "
                         "'CNNAttentionResidual', 'ResNet_mcmc'")
    return net


class StepperParameters:
    """Dataclass for stepper parameters"""
    def __init__(
            self,
            stepper_type: str,
            dt: float,
            relative_tol: float,
            max_step: float,
            min_step: float,
            bulirsch_k_min: int,
            bulirsch_k_max: int
    ):
        self.stepper_type = stepper_type
        self.dt = dt
        self.relative_tol = relative_tol
        self.max_step = max_step
        self.min_step = min_step
        self.bulirsch_k_min = bulirsch_k_min
        self.bulirsch_k_max = bulirsch_k_max


def get_stepper(
        step_parameters: StepperParameters
) -> Union[jvmc_utilities.stepper.BulirschStoer, jVMC.util.Euler, jVMC.util.AdaptiveHeun,
           jvmc_utilities.stepper.minAdaptiveHeun]:
    if step_parameters.stepper_type == "BulirschStoer":
        stepper = jvmc_utilities.stepper.BulirschStoer(timeStep=step_parameters.dt,
                                                       rtol=step_parameters.integrateTol,
                                                       maxStep=step_parameters.max_step,
                                                       kmin=step_parameters.bulirsch_k_min,
                                                       kmax=step_parameters.bulirsch_k_max)
    elif step_parameters.stepper_type == "Heun":
        stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=step_parameters.dt,
                                                 tol=step_parameters.relative_tol,
                                                 maxStep=step_parameters.max_step)
    elif step_parameters.stepper_type == "minHeun":
        stepper = jvmc_utilities.stepper.minAdaptiveHeun(timeStep=step_parameters.dt,
                                                         tol=step_parameters.relative_tol,
                                                         maxStep=step_parameters.max_step,
                                                         minStep=step_parameters.min_step)
    else:
        if step_parameters.stepper_type != "Euler":
            print("Invalid stepper type. Falling back to Euler.")
        stepper = jVMC.util.stepper.Euler(timeStep=step_parameters.dt)

    return stepper


def aqi_povm_object(dim: str, system_size: int) -> jVMC.operator.POVM:
    povm = jVMC.operator.POVM({"dim": dim, "L": 2 * system_size})
    jvmc_utilities.operators.initialisation_operators(povm)
    jvmc_utilities.operators.aqi_model_operators(povm)
    return povm


def argument_parser() -> argparse.ArgumentParser:
    """
    Returns unparsed ArgumentParser object containing standard parameters.

    Add further parameters using parser.add_argument("--argument", type=type, default=default_value).

    ArgumentParser can be parsed using parser.parse_args().

    The following arguments are already included:

    Inital state parameters: 'eps_0', 'eps_1', 'eps_2'
    Physical parameters: 'L', 'gamma', 'h_transverse', 'Tmax'
    Network parameters: 'network', 'param_dtype', 'depth', 'features', 'attention_heads', 'kernel_size',
                        'embeddingDimFac', 'symmetry'
    Stepper parameters: 'stepper', 'dt', 'integrateTol', 'maxStep', 'minStep', 'bulirschkmin', 'bulirschkmax'
    Sampler parameters: 'use_exact_sampler', 'numSamples', 'mu', 'numChains', 'thermalizationSweeps',
                        'sweepStepsMultiplier'
    jVMC parameters:  'seed', 'measSamples', 'measSteps', 'batchSize', 'ElocBatchSize', 'pinvTol', 'pinvCutoff',
                      'snrTol', 'diagonalizeOnDevice', 'crossValidation'
    further parameters: 'start_from_last'
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Process some parameters.')

    # Intial state parameters
    parser.add_argument('--eps_0', type=float, default=0.01)
    parser.add_argument('--eps_1', type=float, default=0.4)
    parser.add_argument('--eps_2', type=float, default=0.1)

    # Physical parameters
    parser.add_argument('--L', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--h_transverse', type=float, default=1.0)
    parser.add_argument('--Tmax', type=float, default=5.0)

    # Network parameters
    parser.add_argument('--network', type=str, default='DeepNADE',
                        choices=["DeepNADE", "GatedCNN", "CNN", "RNN", "LSTM", "AFFN", "CNNResidual", "CNNAttention",
                                 "CNNAttentionResidual", "CNN_mcmc", "CNNEmbedded", "CNNEmbeddedResidual",
                                 "ResNet_mcmc"])
    parser.add_argument('--param_dtype', type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--features', type=int, default=4)
    parser.add_argument('--attention_heads', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=1)
    parser.add_argument('--embeddingDimFac', type=int, default=1)
    parser.add_argument('--symmetry', type=str2bool, default=False)

    # Stepper parameters
    parser.add_argument('--stepper', type=str, default="Heun", choices=["Euler", "Heun", "BulirschStoer"])
    parser.add_argument('--integrateTol', type=float, default=1e-7)
    parser.add_argument('--dt', type=float, default=1e-3)
    parser.add_argument('--maxStep', type=float, default=1)
    parser.add_argument('--minStep', type=float, default=1e-5)
    parser.add_argument('--bulirschkmin', type=int, default=2)
    parser.add_argument('--bulirschkmax', type=int, default=8)

    # Sampler parameters
    parser.add_argument('--use_exact_sampler', type=str2bool, default=False)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--numChains', type=int, default=100)
    parser.add_argument('--sweepStepsMultiplier', type=int, default=3)
    parser.add_argument('--thermalizationSweeps', type=int, default=8)
    parser.add_argument('--numSamples', type=int, default=10000)

    # jVMC parameters
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--measSamples', type=int, default=10000)
    parser.add_argument('--measSteps', type=int, default=None)
    parser.add_argument('--batchSize', type=int, default=10000)
    parser.add_argument('--ElocBatchSize', type=int, default=-1)
    parser.add_argument('--pinvTol', type=float, default=1e-7)
    parser.add_argument('--pinvCutoff', type=float, default=1e-8)
    parser.add_argument('--snrTol', type=float, default=2)
    parser.add_argument('--diagonalizeOnDevice', type=str2bool, default=False)
    parser.add_argument('--crossValidation', type=str2bool, default=False)

    # further parameters
    parser.add_argument('--start_from_last', type=bool, default=False)
    return parser


class TDVP_Norm:
    """
    Helper class to implement normalisation function for TDVP equation.
    """
    def __init__(self, tdvpEquation: jVMC.util.TDVP):
        self.tdvpEquation = tdvpEquation

    def norm_function(self,  v: jnp.ndarray) -> float:
        return jnp.real(jnp.conj(jnp.transpose(v)).dot(self.tdvpEquation.S_dot(v)))


class H5PY_wrapper:
    """
    Wrapper class around a h5py file to write data and metadata specific to jvmc_utility tasks.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

        self._has_metadata = None

        if mpi.rank == 0:
            with h5py.File(self.file_path, mode="a") as f:
                self._has_metadata = 0 != len(f.attrs.keys())
                try:
                    f.create_group("network_checkpoints/")
                    f.create_group("sampler_checkpoints/")
                    f.create_group("observables/")
                except ValueError:
                    pass
        self._has_metadata = mpi.comm.bcast(self._has_metadata, root=0)

    def has_metadata(self) -> bool:
        return self._has_metadata

    def write_metadata(self, **kwargs):
        if mpi.rank == 0:
            with h5py.File(self.file_path, "a") as f:
                for key, value in kwargs.items():
                    f.attrs[key] = value

    def update_dataset(self, file, group, value):
        newLen = len(file[group]) + 1
        file[group].resize(newLen, axis=0)
        file[group][-1] = value

    def write_observables(self, time, **kwargs):
        if mpi.rank == 0:
            with h5py.File(self.file_path, mode="a") as f:
                if "observables/time" not in f:
                    f.create_dataset("observables/time", (0,), maxshape=(None,), dtype="f8", chunks=True)

                self.update_dataset(f, "observable/time", time)

                for key, value in kwargs.items():
                    if "observables/" + key not in f:
                        f.create_dataset("observables/" + key, (0,), maxshape=(None,), dtype="f8", chunks=True)

                    self.update_dataset(f, "observables/" + key, value)

    def write_network_checkpoint(self, time, parameters):
        if mpi.rank == 0:
            with h5py.File(self.file_path, mode="a") as f:
                if "network_checkpoints/checkpoints" not in f:
                    f.create_dataset("network_checkpoints/checkpoints", shape=(0,) + parameters.shape, dtype="f8",
                                     chunks=True, maxshape=(None,) + parameters.shape)
                if "network_checkpoints/times" not in f:
                    f.create_dataset("network_checkpoints/times", shape=(0,), dtype="f8", chunks=True, maxshape=(None,))

                self.update_dataset(f, "network_checkpoints/time", time)

                self.update_dataset(f, "network_checkpoints/checkpoints", parameters)

    def write_sampler_checkpoint(self, time, state):
        if mpi.rank == 0:
            with h5py.File(self.file_path, mode="a") as f:
                if "sampler_checkpoint/time" not in f:
                    f.create_dataset("sampler_checkpoint/time", shape=(0,), dtype="f8", maxshape=(None,), chunks=True)
                if "sampler_checkpoint/states" not in f:
                    f.create_dataset("sampler_checkpoint/states", dtype="i1", chunks=True,
                                     shape=(0,) + state.shape, maxshape=(None,) + state.shape)

                self.update_dataset(f, "sampler_checkpoint/time", time)
                self.update_dataset(f, "sampler_checkpoint/states", state)

    def print(self, message):
        if mpi.rank == 0:
            print(message, flush=True)
