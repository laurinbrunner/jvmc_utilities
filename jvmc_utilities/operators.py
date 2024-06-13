import jax
import jax.numpy as jnp
import jVMC.operator as jvmcop
import jVMC.global_defs as global_defs
import jVMC.mpi_wrapper as mpi
import itertools
from jVMC.util import TDVP
from jVMC.stats import SampledObs
from typing import Tuple


def initialisation_operators(povm: jvmcop.POVM) -> None:
    """
    Extends operators in POVM object by operators useful for state initialisation.

    Adds two-site dissipative operators "upup_dis", "updown_dis", "downup_dis" and "downdown_dis", which have an up-up,
    up-down, down-up and down-down steady state respectively. Also adds single site dissipative operators "up_dis" and
    "down_dis" that have an up and down steady state.
    """
    M_2Body, T_inv_2Body = higher_order_M_T_inv(2, povm.M, povm.T_inv)

    if "up_dis" not in povm.operators.keys():
        up = jnp.array([[0, 1], [0, 0]])
        povm.add_dissipator("up_dis", jvmcop.matrix_to_povm(up, povm.M, povm.T_inv, mode="dis"))

    if "down_dis" not in povm.operators.keys():
        down = jnp.array([[0, 0], [1, 0]])
        povm.add_dissipator("down_dis", jvmcop.matrix_to_povm(down, povm.M, povm.T_inv, mode="dis"))

    if "upup_dis" not in povm.operators.keys():
        uu1 = jnp.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        uu2 = jnp.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        uu3 = jnp.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        povm.add_dissipator("upup_dis", (jvmcop.matrix_to_povm(uu1, M_2Body, T_inv_2Body, mode="dis") +
                                         jvmcop.matrix_to_povm(uu2, M_2Body, T_inv_2Body, mode="dis") +
                                         jvmcop.matrix_to_povm(uu3, M_2Body, T_inv_2Body, mode="dis")))

    if "updown_dis" not in povm.operators.keys():
        ud1 = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        ud2 = jnp.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        ud3 = jnp.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        povm.add_dissipator("updown_dis", (jvmcop.matrix_to_povm(ud1, M_2Body, T_inv_2Body, mode="dis") +
                                           jvmcop.matrix_to_povm(ud2, M_2Body, T_inv_2Body, mode="dis") +
                                           jvmcop.matrix_to_povm(ud3, M_2Body, T_inv_2Body, mode="dis")))

    if "downup_dis" not in povm.operators.keys():
        du1 = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        du2 = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        du3 = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        povm.add_dissipator("downup_dis", (jvmcop.matrix_to_povm(du1, M_2Body, T_inv_2Body, mode="dis") +
                                           jvmcop.matrix_to_povm(du2, M_2Body, T_inv_2Body, mode="dis") +
                                           jvmcop.matrix_to_povm(du3, M_2Body, T_inv_2Body, mode="dis")))

    if "downdown_dis" not in povm.operators.keys():
        dd1 = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        dd2 = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        dd3 = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
        povm.add_dissipator("downdown_dis", (jvmcop.matrix_to_povm(dd1, M_2Body, T_inv_2Body, mode="dis") +
                                             jvmcop.matrix_to_povm(dd2, M_2Body, T_inv_2Body, mode="dis") +
                                             jvmcop.matrix_to_povm(dd3, M_2Body, T_inv_2Body, mode="dis")))


def higher_order_M_T_inv(order: int, M: jnp.ndarray, T_inv: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns POVM observables and inverse of overlap matrix of higher order.

    :param order:
    :param M:
    :return:
    """
    if type(order) != int or order < 1:
        raise ValueError("order must be an integer greater than 0.")
    if order == 1:
        return M, T_inv

    _M = []
    for indices in itertools.product([0, 1, 2, 3], repeat=order):
        helper = jnp.kron(M[indices[0]], M[indices[1]])
        for i in range(2, order):
            helper = jnp.kron(helper, M[indices[i]])
        _M.append(helper)

    _M = jnp.array(_M).reshape(4**order, 2**order, 2**order)

    _T = jnp.kron(T_inv, T_inv)
    for i in range(2, order):
        _T = jnp.kron(_T, T_inv)

    return _M, _T


def aqi_model_operators(povm: jvmcop.POVM) -> None:
    """
    Adds operators used in the active quantum ising model to the POVM object.

    This function adds the operator :math:`\sigma^+\sigma^-$` in both unitary and dissipative form and the operator
    :math:`\sigma^z\sigma^+\sigma^-` in dissipative form.
    They are called "spin_flip_uni", "spin_flip_dis" and "spin_flip_Z_dis" respectively.
    """
    M_2Body, T_inv_2Body = higher_order_M_T_inv(2, povm.M, povm.T_inv)
    M_3Body, T_inv_3Body = higher_order_M_T_inv(3, povm.M, povm.T_inv)
    M_4Body, T_inv_4Body = higher_order_M_T_inv(4, povm.M, povm.T_inv)

    sigmas = jvmcop.get_paulis()

    # The following are spin-1/2 ladder operators for the \sigma_z, \sigma_x and \sigma_y basis respectively
    sz_plus = (sigmas[0] + 1j * sigmas[1]) / 2
    sz_minus = (sigmas[0] - 1j * sigmas[1]) / 2

    spin_flip = jnp.kron(sz_plus, sz_minus)
    spin_flip_Z = jnp.kron(jnp.kron(sz_plus, sz_minus), sigmas[2])
    inv_spin_flip = jnp.kron(sz_minus, sz_plus)
    inv_spin_flip_Z = jnp.kron(jnp.kron(sz_minus, sz_plus), sigmas[2])
    restricted_jump = jnp.kron(jnp.kron(sz_plus, jnp.eye(2) - sigmas[2]), sz_minus) / 2
    conditional_cluster_pmnn = jnp.kron(jnp.kron(jnp.kron(sz_plus, sz_minus), jnp.eye(2) + sigmas[2]),
                                        jnp.eye(2) + sigmas[2]) / 4

    if "spin_flip_uni" not in povm.operators.keys():
        povm.add_unitary("spin_flip_uni", jvmcop.matrix_to_povm(spin_flip, M_2Body, T_inv_2Body, mode="uni"))
    if "spin_flip_dis" not in povm.operators.keys():
        povm.add_dissipator("spin_flip_dis", jvmcop.matrix_to_povm(spin_flip, M_2Body, T_inv_2Body, mode="dis"))
    if "spin_flip_Z_dis" not in povm.operators.keys():
        povm.add_dissipator("spin_flip_Z_dis", jvmcop.matrix_to_povm(spin_flip_Z, M_3Body, T_inv_3Body, mode="dis"))
    if "inv_spin_flip_uni" not in povm.operators.keys():
        povm.add_unitary("inv_spin_flip_uni", jvmcop.matrix_to_povm(inv_spin_flip, M_2Body, T_inv_2Body, mode="uni"))
    if "inv_spin_flip_dis" not in povm.operators.keys():
        povm.add_dissipator("inv_spin_flip_dis", jvmcop.matrix_to_povm(inv_spin_flip, M_2Body, T_inv_2Body, mode="dis"))
    if "inv_spin_flip_Z_dis" not in povm.operators.keys():
        povm.add_dissipator("inv_spin_flip_Z_dis", jvmcop.matrix_to_povm(inv_spin_flip_Z, M_3Body, T_inv_3Body,
                                                                         mode="dis"))
    if "restricted_jump_dis" not in povm.operators.keys():
        povm.add_dissipator("restricted_jump_dis", jvmcop.matrix_to_povm(restricted_jump, M_3Body, T_inv_3Body,
                                                                         mode="dis"))
    if "cond_cluster_pmnn_dis" not in povm.operators.keys():
        povm.add_dissipator("cond_cluster_pmnn_dis", jvmcop.matrix_to_povm(conditional_cluster_pmnn, M_4Body,
                                                                           T_inv_4Body, mode="dis"))


class EfficientPOVMOperator(jvmcop.POVMOperator):
    """
    More efficient implementation of the POVMOperator class, that only evaluates the s_prime configurations that have a
    non-zero weight.
    """

    def get_O_loc(self, samples, psi, logPsiS=None, *args):
        """Compute :math:`O_{loc}(s)`.

        If the instance parameter ElocBatchSize is larger than 0 :math:`O_{loc}(s)` is computed in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math:`O_{loc}(s)` for each configuration :math:`s`.
        """

        if logPsiS is None:
            logPsiS = psi(samples)

        if self.ElocBatchSize > 0:
            return self.get_O_loc_batched(samples, psi, logPsiS, self.ElocBatchSize, *args)
        else:
            sp, matEl = self.get_s_primes(samples, *args)

            logPsiSP = jnp.zeros((sp.shape[0], sp.shape[1]), dtype=logPsiS.dtype)
            nonzero_idx = jnp.where(jnp.logical_not(jnp.isclose(matEl.reshape(matEl.shape[0], -1), 0)))
            logPsiSP = logPsiSP.at[nonzero_idx].set(psi(sp[nonzero_idx].reshape(1, -1, sp.shape[-1])).reshape(-1))

            return self.get_O_loc_unbatched(logPsiS, logPsiSP)

    def get_O_loc_batched(self, samples, psi, logPsiS, batchSize, *args):
        """Compute :math:`O_{loc}(s)` in batches.

        Computes :math:`O_{loc}(s)=\sum_{s'} O_{s,s'}\\frac{\psi(s')}{\psi(s)}` in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``batchSize``: Batch size.
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math:`O_{loc}(s)` for each configuration :math:`s`.
        """

        Oloc = None

        numSamples = samples.shape[1]
        numBatches = numSamples // batchSize
        remainder = numSamples % batchSize

        # Minimize mismatch
        if remainder > 0:
            batchSize = numSamples // (numBatches + 1)
            numBatches = numSamples // batchSize
            remainder = numSamples % batchSize

        for b in range(numBatches):

            batch = self._get_config_batch_pmapd(samples, b * batchSize, batchSize)
            logPsiSbatch = self._get_logPsi_batch_pmapd(logPsiS, b * batchSize, batchSize)

            sp, matEl = self.get_s_primes(batch, *args)

            logPsiSP = jnp.zeros((sp.shape[0], sp.shape[1]), dtype=logPsiS.dtype)
            nonzero_idx = jnp.where(jnp.logical_not(jnp.isclose(matEl.reshape(matEl.shape[0], -1), 0)))
            logPsiSP = logPsiSP.at[nonzero_idx].set(psi(sp[nonzero_idx].reshape(1, -1, sp.shape[-1])).reshape(-1))

            OlocBatch = self.get_O_loc_unbatched(logPsiSbatch, logPsiSP)

            if Oloc is None:
                if OlocBatch.dtype == global_defs.tCpx:
                    Oloc = self._alloc_Oloc_cpx_pmapd(samples)
                else:
                    Oloc = self._alloc_Oloc_real_pmapd(samples)

            Oloc = self._insert_Oloc_batch_pmapd(Oloc, OlocBatch, b * batchSize)

        if remainder > 0:
            batch = self._get_config_batch_pmapd(samples, numBatches * batchSize, remainder)
            batch = global_defs.pmap_for_my_devices(jvmcop.expand_batch, static_broadcasted_argnums=(1,))(batch,
                                                                                                          batchSize)
            logPsiSbatch = self._get_logPsi_batch_pmapd(logPsiS, numBatches * batchSize, numSamples % batchSize)
            logPsiSbatch = global_defs.pmap_for_my_devices(jvmcop.expand_batch, static_broadcasted_argnums=(1,))(
                logPsiSbatch, batchSize)

            sp, matEl = self.get_s_primes(batch, *args)

            logPsiSP = jnp.zeros((sp.shape[0], sp.shape[1]), dtype=logPsiS.dtype)
            nonzero_idx = jnp.where(jnp.logical_not(jnp.isclose(matEl.reshape(matEl.shape[0], -1), 0)))
            logPsiSP = logPsiSP.at[nonzero_idx].set(psi(sp[nonzero_idx].reshape(1, -1, sp.shape[-1])).reshape(-1))

            OlocBatch = self.get_O_loc_unbatched(logPsiSbatch, logPsiSP)

            OlocBatch = self._get_Oloc_slice_pmapd(OlocBatch, 0, remainder)

            Oloc = self._insert_Oloc_batch_pmapd(Oloc, OlocBatch, numBatches * batchSize)

        return Oloc


class TDVP_with_constraint(TDVP):
    """
    Class for calculating parameter updates from TDVP equation with an additional constraint in the POVM framework.

    This class inherits from jVMC.util.TDVP and works similar to it in almost all aspects.
    The `omega` parameter should be the POVM representation of an observable that has an expectation value of 0
    throughout the time evolution. `k` vives the number of sites the observable 'looks' at.
    """

    def __init__(self, sampler, omega, k, **kwargs):
        super().__init__(sampler, **kwargs)
        self._omega = omega
        if k == 1:
            self.omega_k = self._omega_1
        elif k == 2:
            self.omega_k = self._omega_2
        else:
            raise ValueError("Only implemented values for k are 1 and 2.")
        self.c_over_time = []

    def _omega_2(self, samples):
        s = 4 * samples[..., ::2] + samples[..., 1::2]
        return jnp.sum(self._omega[s], axis=-1)

    def _omega_1(self, samples):
        return jnp.sum(self._omega[samples], axis=-1)

    def __call__(self, netParameters, t, *, psi, hamiltonian, **rhsArgs):
        """ For given network parameters this function solves the TDVP equation.

        This function returns :math:`\\dot\\theta=S^{-1}F`. Thereby an instance of the ``TDVP`` class is a suited
        callable for the right hand side of an ODE to be used in combination with the integration schemes
        implemented in ``jVMC.stepper``. Alternatively, the interface matches the scipy ODE solvers as well.

        Arguments:
            * ``netParameters``: Parameters of the NQS.
            * ``t``: Current time.
            * ``psi``: NQS ansatz. Instance of ``jVMC.vqs.NQS``.
            * ``hamiltonian``: Hamiltonian operator, i.e., an instance of a derived class of ``jVMC.operator.Operator``. \
                                *Notice:* Current time ``t`` is by default passed as argument when computing matrix elements.

        Further optional keyword arguments:
            * ``numSamples``: Number of samples to be used by MC sampler.
            * ``outp``: An instance of ``jVMC.OutputManager``. If ``outp`` is given, timings of the individual steps \
                are recorded using the ``OutputManger``.
            * ``intStep``: Integration step number of multi step method like Runge-Kutta. This information is used to store \
                quantities like energy or residuals at the initial integration step.

        Returns:
            The solution of the TDVP equation, :math:`\\dot\\theta=S^{-1}F`.
        """

        tmpParameters = psi.get_parameters()
        psi.set_parameters(netParameters)

        outp = None
        if "outp" in rhsArgs:
            outp = rhsArgs["outp"]
        self.outp = outp

        numSamples = None
        if "numSamples" in rhsArgs:
            numSamples = rhsArgs["numSamples"]

        def start_timing(outp, name):
            if outp is not None:
                outp.start_timing(name)

        def stop_timing(outp, name, waitFor=None):
            if waitFor is not None:
                waitFor.block_until_ready()
            if outp is not None:
                outp.stop_timing(name)

        # Get sample
        start_timing(outp, "sampling")
        sampleConfigs, sampleLogPsi, p = self.sampler.sample(numSamples=numSamples)
        stop_timing(outp, "sampling", waitFor=sampleConfigs)

        # Evaluate local energy
        start_timing(outp, "compute Eloc")
        Eloc = hamiltonian.get_O_loc(sampleConfigs, psi, sampleLogPsi, t)
        omega = self.omega_k(sampleConfigs)
        omega_samob = SampledObs(omega, p)
        Eloc_samob = SampledObs(Eloc, p)
        c = - Eloc_samob.covar(omega_samob).ravel()[0] / omega_samob.var()[0]
        self.c_over_time.append(c)
        stop_timing(outp, "compute Eloc", waitFor=Eloc)
        Eloc = SampledObs(Eloc + c * omega, p)

        # Evaluate gradients
        start_timing(outp, "compute gradients")
        sampleGradients = psi.gradients(sampleConfigs)
        stop_timing(outp, "compute gradients", waitFor=sampleGradients)
        sampleGradients = SampledObs(sampleGradients, p)

        start_timing(outp, "solve TDVP eqn.")
        update, solverResidual = self.solve(Eloc, sampleGradients)
        stop_timing(outp, "solve TDVP eqn.")

        if outp is not None:
            outp.add_timing("MPI communication", mpi.get_communication_time())

        psi.set_parameters(tmpParameters)

        if "intStep" in rhsArgs:
            if rhsArgs["intStep"] == 0:

                self.ElocMean0 = self.ElocMean
                self.ElocVar0 = self.ElocVar

                self.metaData = {
                    "tdvp_error": self._get_tdvp_error(update),
                    "tdvp_residual": solverResidual,
                    "SNR": self.snr,
                    "spectrum": self.ev,
                }

                if self.crossValidation:
                    Eloc1 = Eloc.subset(start=0, step=2)
                    sampleGradients1 = sampleGradients.subset(start=0, step=2)
                    Eloc2 = Eloc.subset(start=1, step=2)
                    sampleGradients2 = sampleGradients.subset(start=1, step=2)
                    update_1, _ = self.solve(Eloc1, sampleGradients1)
                    S2, F2 = self.get_tdvp_equation(Eloc2, sampleGradients2)

                    validation_tdvpErr = self._get_tdvp_error(update_1)
                    update, solverResidual = self.solve(Eloc, sampleGradients)
                    validation_residual = (jnp.linalg.norm(S2.dot(update_1) - F2) / jnp.linalg.norm(
                        F2)) / solverResidual

                    self.crossValidationFactor_residual = validation_residual
                    self.crossValidationFactor_tdvpErr = validation_tdvpErr / self.metaData["tdvp_error"]
                    self.metaData["tdvp_residual_cross_validation_ratio"] = self.crossValidationFactor_residual
                    self.metaData["tdvp_error_cross_validation_ratio"] = self.crossValidationFactor_tdvpErr

                    self.S, _ = self.get_tdvp_equation(Eloc, sampleGradients)

        return update
