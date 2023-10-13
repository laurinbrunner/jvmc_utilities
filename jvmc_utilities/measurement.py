import jax.numpy as jnp
import jVMC.sampler
import jVMC.operator as jvmcop
import jVMC.mpi_wrapper as mpi
from .operators import higher_order_M_T_inv
from typing import Union, List, Dict, Tuple


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

        M_2_body, T_inv_2_body = higher_order_M_T_inv(2, self.povm.M, self.povm.T_inv)
        M_3_body, T_inv_3_body = higher_order_M_T_inv(3, self.povm.M, self.povm.T_inv)

        sigma_z = jnp.array([[1, 0], [0, -1]])
        n_mat = (sigma_z + jnp.eye(2)) / 2

        self.n_obser = jvmcop.matrix_to_povm(n_mat, self.povm.M, self.povm.T_inv, mode="obs")
        self.n_sq_obser = jvmcop.matrix_to_povm(n_mat @ n_mat, self.povm.M, self.povm.T_inv, mode="obs")
        self.n_corr_obser = jvmcop.matrix_to_povm(jnp.kron(n_mat, n_mat), M_2_body, T_inv_2_body,
                                                  mode="obs").reshape(4, 4)
        self.j_restricted_obser = jvmcop.matrix_to_povm(jnp.kron(jnp.kron(jnp.eye(2) - n_mat, jnp.eye(2) - n_mat),
                                                                 n_mat),
                                                        M_3_body, T_inv_3_body, mode="obs").reshape(4, 4, 4)

        self.observables = []
        self.observables_functions = {"N": self._measure_N, "Sx_i": self._measure_Sx_i, "Sy_i": self._measure_Sy_i,
                                      "Sz_i": self._measure_Sz_i, "M_sq": self._measure_M_sq, "N_i": self._measure_N_i,
                                      "m_corr": self._measure_m_corr, "n_corr": self._measure_n_corr,
                                      "j_restricted": self._measure_j_restricted}
        self.observables_functions_errors = {"N_MC_error": self._measure_N_MC_error,
                                             "Sx_i_MC_error": self._measure_Sx_i_MC_error,
                                             "Sy_i_MC_error": self._measure_Sy_i_MC_error,
                                             "Sz_i_MC_error": self._measure_Sz_i_MC_error,
                                             "N_i_MC_error": self._measure_N_i_MC_error}
        self.calculated_n = False
        self.calculated_n_corr = False

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

    def _calculate_n_corr(self) -> None:
        self.n_corr = jnp.zeros((2 * self.L, 2 * self.L))
        for i in range(2 * self.L):
            for j in range(i, 2 * self.L):
                if i == j:
                    self.n_corr = self.n_corr.at[i, i].set(mpi.global_mean(self.n_sq_obser[self.confs[:, :, i]],
                                                                           self.probs))
                else:
                    corr = mpi.global_mean(self.n_corr_obser[self.confs[:, :, i], self.confs[:, :, j]],
                                           self.probs)
                    self.n_corr = self.n_corr.at[i, j].set(corr)
                    self.n_corr = self.n_corr.at[j, i].set(corr)
        self.calculated_n_corr = True

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
        if not self.calculated_n_corr:
            self._calculate_n_corr()
        return jnp.mean(self.n_corr[::2, ::2] + self.n_corr[1::2, 1::2] - self.n_corr[::2, 1::2]
                        - self.n_corr[1::2, ::2])

    def _measure_m_corr(self) -> jnp.ndarray:
        if not self.calculated_n_corr:
            self._calculate_n_corr()
        return self.n_corr[::2, ::2] + self.n_corr[1::2, 1::2] - self.n_corr[::2, 1::2] - self.n_corr[1::2, ::2]

    def _measure_n_corr(self) -> jnp.ndarray:
        if not self.calculated_n_corr:
            self._calculate_n_corr()
        return self.n_corr

    def _measure_j_restricted(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (jnp.array([mpi.global_mean(self.j_restricted_obser[self.confs[:, :, 2*l], self.confs[:, :, 2*l+1],
                                                                  self.confs[:, :, (2*l+2) % (2*self.L)]], self.probs)
                          for l in range(self.L)]),\
               jnp.array([mpi.global_mean(self.j_restricted_obser[self.confs[:, :, (2*l+2) % (2*self.L)],
                                                                  self.confs[:, :, (2*l+3) % (2*self.L)],
                                                                  self.confs[:, :, (2*l+1) % (2*self.L)]], self.probs)
                          for l in range(self.L)]))

    def measure(self) -> Dict[str, jnp.ndarray]:
        """
        Returns dictionary of measurements.
        """
        self.confs, _, self.probs = self.sampler.sample()

        self.calculated_n = False
        self.calculated_n_corr = False
        results = {}
        for obs in self.observables:
            results[obs] = self.observables_functions[obs]()
        if self.mc_errors:
            for obs in self.observables:
                if obs in ["M_sq", "m_corr", "n_corr"]:
                    continue
                obs = obs + "_MC_error"
                results[obs] = self.observables_functions_errors[obs]()

        # Free up space
        self.confs, self.probs = None, None

        return results
