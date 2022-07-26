import jax.numpy as jnp
import jVMC.sampler
import jVMC.operator as jvmcop
import jVMC.mpi_wrapper as mpi
from .operators import higher_order_M_T_inv
from typing import Union, List, Dict


class Measurement:
    """
    This class provides functionality to measure different observables on a POVM state.

    The supported measurement observables are "Sx_i", "Sy_i", "Sz_i", "N", "N_i" and "M_sq" (which is defined by
    :math:`M^2 = 1/L^2 (\sum_l m_l)^2 `), where the subscribt i indicates site resolved measurements (Note that the
    sites here are not the physical ones but only the computational sites). The "N" measurement is returned as an array
    containing "N_up" and "N_down", whereas "N_i" gives the
    occupation of every computational site.

    Only those observables given through the set_observables method will be calculated and returned.
    """

    def __init__(
            self,
            sampler: Union[jVMC.sampler.MCSampler, jVMC.sampler.ExactSampler],
            povm: jvmcop.POVM,
            mc_errors: bool = False
    ) -> None:
        self.sampler = sampler
        self.povm = povm
        self.L = self.povm.system_data["L"] // 2
        self.mc_errors = mc_errors

        self.M_2_body, self.T_inv_2_body = higher_order_M_T_inv(2, self.povm.M, self.povm.T_inv)

        sigma_z = jnp.array([[1, 0], [0, -1]])
        n_mat = (sigma_z + jnp.eye(2)) / 2

        self.n_obser = jvmcop.matrix_to_povm(n_mat, self.povm.M, self.povm.T_inv, mode="obs")
        self.n_sq_obser = jvmcop.matrix_to_povm(n_mat @ n_mat, self.povm.M, self.povm.T_inv, mode="obs")
        self.n_corr_obser = jvmcop.matrix_to_povm(jnp.kron(n_mat, n_mat), self.M_2_body, self.T_inv_2_body,
                                                  mode="obs").reshape(4, 4)

        self.observables = []
        self.observables_functions = {"N": self._measure_N, "Sx_i": self._measure_Sx_i, "Sy_i": self._measure_Sy_i,
                                      "Sz_i": self._measure_Sz_i, "M_sq": self._measure_M_sq, "N_i": self._measure_N_i,
                                      "m_corr": self._measure_m_corr}
        self.observables_functions_errors = {"N_MC_error": self._measure_N_MC_error,
                                             "Sx_i_MC_error": self._measure_Sx_i_MC_error,
                                             "Sy_i_MC_error": self._measure_Sy_i_MC_error,
                                             "Sz_i_MC_error": self._measure_Sz_i_MC_error,
                                             "N_i_MC_error": self._measure_N_i_MC_error}
        self.calculated_n = False

    def set_observables(self, observables: List[str]) -> None:
        """
        Set observables to measure.

        :param observables: list of names of the observables (for a full list of possible names see docstring of class)
        """
        self.observables = set(observables)

    def _calculate_n(self) -> None:
        self.n = mpi.global_mean(self.n_obser[self.confs], self.probs)
        if self.mc_errors:
            self.n_mc_error = mpi.global_variance(self.n_obser[self.confs], self.probs) / \
                              jnp.sqrt(self.sampler.get_last_number_of_samples())
        self.calculated_n = True

    def _measure_N_i(self) -> jnp.ndarray:
        if not self.calculated_n:
            self._calculate_n()
        return self.n

    def _measure_N_i_MC_error(self) -> jnp.ndarray:
        if not self.calculated_n:
            self._calculate_n()
        return self.n_mc_errors

    def _measure_N(self) -> jnp.ndarray:
        if not self.calculated_n:
            self._calculate_n()
        return jnp.array([jnp.mean(self.n[::2]), jnp.mean(self.n[1::2])])

    def _measure_N_MC_error(self) -> jnp.ndarray:
        if not self.calculated_n:
            self._calculate_n()
        return jnp.array([jnp.mean(self.n_mc_error[::2]), jnp.mean(self.n_mc_error[1::2])])

    def _measure_Sx_i(self) -> jnp.ndarray:
        return mpi.global_mean(self.povm.observables["X"][self.confs], self.probs)

    def _measure_Sx_i_MC_error(self) -> jnp.ndarray:
        return mpi.global_variance(self.povm.observables["X"][self.confs], self.probs) / \
                              jnp.sqrt(self.sampler.get_last_number_of_samples())

    def _measure_Sy_i(self) -> jnp.ndarray:
        return mpi.global_mean(self.povm.observables["Y"][self.confs], self.probs)

    def _measure_Sy_i_MC_error(self) -> jnp.ndarray:
        return mpi.global_variance(self.povm.observables["Y"][self.confs], self.probs) / \
                              jnp.sqrt(self.sampler.get_last_number_of_samples())

    def _measure_Sz_i(self) -> jnp.ndarray:
        return mpi.global_mean(self.povm.observables["Z"][self.confs], self.probs)

    def _measure_Sz_i_MC_error(self) -> jnp.ndarray:
        return mpi.global_variance(self.povm.observables["Z"][self.confs], self.probs) / \
                              jnp.sqrt(self.sampler.get_last_number_of_samples())

    def _measure_M_sq(self) -> jnp.ndarray:
        n_sq_u = mpi.global_mean(self.n_sq_obser[self.confs[:, :, ::2]], self.probs)
        n_sq_d = mpi.global_mean(self.n_sq_obser[self.confs[:, :, 1::2]], self.probs)

        if self.L != 1:
            n_corr_uu = mpi.global_mean(jnp.sum(jnp.array([self.n_corr_obser[self.confs[:, :, ::2],
                                                                             jnp.roll(self.confs[:, :, ::2], j, axis=2)]
                                                           for j in range(1, self.L)]), axis=0), self.probs)
            n_corr_dd = mpi.global_mean(jnp.sum(jnp.array([self.n_corr_obser[self.confs[:, :, 1::2],
                                                                             jnp.roll(self.confs[:, :, 1::2], j, axis=2)]
                                                           for j in range(1, self.L)]), axis=0), self.probs)
        else:
            n_corr_uu, n_corr_dd = 0., 0.
        n_corr_ud = mpi.global_mean(jnp.sum(jnp.array([self.n_corr_obser[self.confs[:, :, ::2],
                                                                         jnp.roll(self.confs[:, :, 1::2], j, axis=2)]
                                                       for j in range(self.L)]), axis=0), self.probs)
        n_corr_du = mpi.global_mean(jnp.sum(jnp.array([self.n_corr_obser[self.confs[:, :, 1::2],
                                                                         jnp.roll(self.confs[:, :, ::2], j, axis=2)]
                                                       for j in range(self.L)]), axis=0), self.probs)

        return jnp.mean(n_sq_u + n_sq_d + n_corr_uu + n_corr_dd - n_corr_ud - n_corr_du) / self.L

    def _measure_m_corr(self) -> jnp.ndarray:
        lower_triangular = jnp.array([[True if j < i else False
                                       for j in range(self.L)] for i in range(self.L)])

        n_uu = jnp.array([[0 if j < i else mpi.global_mean(self.n_sq_obser[self.confs[:, :, i]], self.probs) if i == j
                           else mpi.global_mean(self.n_corr_obser[self.confs[:, :, i], self.confs[:, :, j]], self.probs)
                           for j in range(0, 2*self.L, 2)] for i in range(0, 2*self.L, 2)])

        n_uu = n_uu.at[lower_triangular].set(n_uu.T[lower_triangular])

        n_dd = jnp.array([[0 if j < i else mpi.global_mean(self.n_sq_obser[self.confs[:, :, i]], self.probs) if i == j
                           else mpi.global_mean(self.n_corr_obser[self.confs[:, :, i], self.confs[:, :, j]], self.probs)
                           for j in range(1, 2*self.L, 2)] for i in range(1, 2*self.L, 2)])

        n_dd = n_dd.at[lower_triangular].set(n_dd.T[lower_triangular])

        n_du = jnp.array([[mpi.global_mean(self.n_corr_obser[self.confs[:, :, i], self.confs[:, :, j]], self.probs)
                           for i in range(1, 2*self.L, 2)] for j in range(0, 2*self.L, 2)])

        return n_uu - n_du - n_du.T + n_dd

    def measure(self) -> Dict[str, jnp.ndarray]:
        """
        Returns dictionary of measurements.
        """
        self.confs, _, self.probs = self.sampler.sample()

        self.calculated_n = False
        results = {}
        for obs in self.observables:
            results[obs] = self.observables_functions[obs]()
        if self.mc_errors:
            for obs in self.observables:
                if obs in ["M_sq", "m_corr"]:
                    continue
                obs = obs + "_MC_error"
                results[obs] = self.observables_functions_errors[obs]()

        # Free up space
        self.confs, self.probs = None, None

        return results
