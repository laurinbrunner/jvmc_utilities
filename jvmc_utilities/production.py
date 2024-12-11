import jax.numpy as jnp
import h5py
import jVMC
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
    param_dtype: str
    depth: int
    features: int
    kernel_size: int
    embeddingDimFac: int
    attention_heads: int
    symmetry: bool


def get_network(
        network_type: str,
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

    if network_type == "CNN":
        net = ju_nets.POVMCNN(L=2*system_size, depth=hyperparameters.depth, features=hyperparameters.features,
                              orbit=orbit, kernel_size=(hyperparameters.kernel_size,), param_dtype=dtype)
    elif network_type == "GatedCNN":
        net = ju_nets.POVMCNNGated(L=2*system_size, depth=hyperparameters.depth, features=hyperparameters.features,
                                   orbit=orbit, kernel_size=(hyperparameters.kernel_size,), param_dtype=dtype)
    elif network_type == "CNNResidual":
        net = ju_nets.POVMCNNResidual(L=2*system_size, depth=hyperparameters.depth, features=hyperparameters.features,
                                      orbit=orbit, kernel_size=(hyperparameters.kernel_size,), param_dtype=dtype)
    elif network_type == "CNNEmbedded":
        net = ju_nets.POVMCNNEmbedded(L=system_size, depth=hyperparameters.depth, features=hyperparameters.features,
                                      orbit=orbit, kernel_size=(hyperparameters.kernel_size,), param_dtype=dtype,
                                      embeddingDimFac=hyperparameters.embeddingDimFac)
    elif network_type == "CNNEmbeddedResidual":
        net = ju_nets.POVMCNNEmbeddedResidual(L=system_size, depth=hyperparameters.depth,
                                              features=hyperparameters.features, orbit=orbit,
                                              kernel_size=(hyperparameters.kernel_size,), param_dtype=dtype,
                                              embeddingDimFac=hyperparameters.embeddingDimFac)
    elif network_type == "CNNAttention":
        net = ju_nets.CNNAttention(L=2*system_size, depth=hyperparameters.depth, features=hyperparameters.features,
                                   orbit=orbit, attention_heads=hyperparameters.attention_heads,
                                   kernel_size=(hyperparameters.kernel_size,), param_dtype=dtype)
    elif network_type == "CNNAttentionResidual":
        net = ju_nets.CNNAttentionResidual(L=2*system_size, depth=hyperparameters.depth,
                                           features=hyperparameters.features, orbit=orbit,
                                           attention_heads=hyperparameters.attention_heads,
                                           kernel_size=(hyperparameters.kernel_size,), param_dtype=dtype)
    elif network_type == "DeepNADE":
        net = ju_nets.DeepNADE(L=2*system_size, depth=hyperparameters.depth, hiddenSize=hyperparameters.features,
                               orbit=orbit)
    elif network_type == "AFFN":
        net = ju_nets.AFFN(L=2*system_size, depth=hyperparameters.depth, hiddenSize=hyperparameters.features,
                           orbit=orbit)
    elif network_type == "RNN":
        if hyperparameters.symmetry:
            net = jVMC.nets.RNN1DGeneralSym(L=2*system_size, depth=hyperparameters.depth,
                                            hiddenSize=hyperparameters.features, orbit=orbit,
                                            logProbFactor=logProbFactor, realValuedOutput=True, inputDim=inputDim,
                                            cell="RNN")
        else:
            net = jVMC.nets.RNN1DGeneral(L=2*system_size, depth=hyperparameters.depth, inputDim=inputDim,
                                         hiddenSize=hyperparameters.features, logProbFactor=logProbFactor,
                                         realValuedOutput=True, cell="RNN")
    elif network_type == "LSTM":
        if hyperparameters.symmetry:
            net = jVMC.nets.RNN1DGeneralSym(L=2*system_size, depth=hyperparameters.depth, inputDim=inputDim,
                                            hiddenSize=hyperparameters.features, orbit=orbit,
                                            logProbFactor=logProbFactor, realValuedOutput=True, cell="LSTM")
        else:
            net = jVMC.nets.RNN1DGeneral(L=2*system_size, depth=hyperparameters.depth, cell="LSTM", inputDim=inputDim,
                                         hiddenSize=hyperparameters.features, logProbFactor=logProbFactor,
                                         realValuedOutput=True)
    elif network_type == "CNN_mcmc":
        net = ju_nets.MCMC_CNN(features=hyperparameters.features, depth=hyperparameters.depth,
                               kernel_size=(hyperparameters.kernel_size,))
    else:
        raise ValueError("Unknown network type. Valid types are 'GatedCNN', 'CNN', 'DeepNADE', 'AFFN', 'RNN', 'LSTM', "
                         "'CNN_mcmc', 'CNNResidual', 'CNNEmbedded', 'CNNEmbeddedResidual', 'CNNAttention', "
                         "'CNNAttentionResidual'")
    return net


class StepperParameters:
    """Dataclass for stepper parameters"""
    stepper_type: str
    dt: float
    relative_tol: float
    max_step: float
    min_step: float
    bulirsch_k_min: int
    bulirsch_k_max: int


def get_stepper(
        step_parameters: StepperParameters
) -> Union[jvmc_utilities.stepper.BulirschStoer, jVMC.util.Euler, jVMC.util.AdaptiveHeun,
           jvmc_utilities.stepper.minAdaptiveHeun]:
    if step_parameters.stepper_type == "BulirschStoer":
        stepper = jvmc_utilities.stepper.BulirschStoer(timeStep=step_parameters.dt, rtol=step_parameters.integrateTol,
                                                       maxStep=step_parameters.max_step,
                                                       kmin=step_parameters.bulirsch_k_min,
                                                       kmax=step_parameters.bulirsch_k_max)
    elif step_parameters.stepper_type == "Heun":
        stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=step_parameters.dt, tol=step_parameters.relative_tol,
                                                 maxStep=step_parameters.max_step)
    elif step_parameters.stepper_type == "minHeun":
        stepper = jvmc_utilities.stepper.minAdaptiveHeun(timeStep=step_parameters.dt, tol=step_parameters.relative_tol,
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
    """Returns ArgumentParser with parsed parameters"""
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
                        choices=["DeepNADE", "GatedCNN", "CNN", "RNN", "LSTM",
                                 "AFFN", "CNNResidual", "CNNAttention",
                                 "CNNAttentionResidual", "CNN_mcmc",
                                 "CNNEmbedded", "CNNEmbeddedResidual"])
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

    # jVMC parameters
    parser.add_argument('--use_exact_sampler', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--numSamples', type=int, default=10000)
    parser.add_argument('--measSamples', type=int, default=10000)
    parser.add_argument('--measSteps', type=int, default=None)
    parser.add_argument('--batchSize', type=int, default=10000)
    parser.add_argument('--ElocBatchSize', type=int, default=-1)
    parser.add_argument('--pinvTol', type=float, default=1e-7)
    parser.add_argument('--pinvCutoff', type=float, default=1e-8)
    parser.add_argument('--snrTol', type=float, default=2)
    parser.add_argument('--diagonalizeOnDevice', type=str2bool, default=False)
    parser.add_argument('--crossValidation', type=str2bool, default=False)

    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--start_from_last', type=bool, default=False)
    parser.add_argument('--mu', type=float, default=1.0)

    parser.add_argument('--direction', type=str, default="None", choices=["increase", "decrease", "size", "None"])
    parser.add_argument('--previous_h', type=float, default=-1.0)

    args = parser.parse_args()