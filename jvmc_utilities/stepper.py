import numpy as np
import jax.numpy
from typing import Tuple


class BulirschStoer:
    """
    This class implements the Bulirsch-Stoer method as described in 'Numerical Recipes: The Art of Scientific Computing,
    Third Edition' Chapter 17.3.
    """

    def __init__(
            self,
            timeStep: float = 1e-2,
            kmax: int = 8,
            kmin: int = 1,
            atol: float = 1e-5,
            rtol: float = 1e-7,
            maxStep: float = 0.1
    ) -> None:
        self.kmax = kmax
        self.kmin = kmin
        self.substeps = 2 * (np.arange(self.kmax+1, dtype=int) + 1)
        self.dt = timeStep
        self.atol = atol
        self.rtol = rtol
        self.maxStep = maxStep

        # S1 and S2 are safety factors smaller than one
        self.S1 = 0.94
        self.S2 = 0.65

        self.compiled_midpoints = {n_k: lambda t, f, y, step_size, **rhsArgs:
                                             self._midpoint(t, f, y, step_size, n_k, **rhsArgs)
                                   for n_k in self.substeps}
        self.A_k = np.array([np.sum(self.substeps[:k]) + 1 for k in range(self.kmax + 1)])
        self.__k_target = self.kmax - 1

    @property
    def k_target(self) -> int:
        return self.__k_target

    @k_target.setter
    def k_target(self, new_k: int) -> None:
        if self.kmin <= new_k <= self.kmax:
            self.__k_target = new_k

    def _midpoint(
            self,
            t: float,
            f: callable,
            y: jax.numpy.ndarray,
            step_size: float,
            k: int,
            **rhsArgs
    ) -> jax.numpy.ndarray:
        h = step_size / k

        z = jax.numpy.zeros((k + 1, y.size), dtype=y.dtype)
        z = z.at[0].set(y)
        z = z.at[1].set(z[0] + h * f(y, t, **rhsArgs, intStep=0))
        for i in range(2, k + 1):
            z = z.at[i].set(z[i - 2] + 2 * h * f(z[i - 1], t + (i-1) * h, **rhsArgs, intStep=i))

        return (z[-1] + z[-2] + h * f(z[-1], t + step_size, **rhsArgs, intStep=k+1)) / 2

    def _polynomial_extrapolation(self, T_matrix: jax.numpy.ndarray, k: int, j: int) -> jax.numpy.ndarray:
        return T_matrix[k, j] + (T_matrix[k, j] - T_matrix[k-1, j])/((self.substeps[k]/self.substeps[k-j-1])**2 - 1)

    def _estimated_step(self, dt: float, err: float, k: int) -> float:
        return dt * self.S1 * (self.S2 / err)**(1/(2*k-1))

    def step(
            self,
            t: float,
            f: callable,
            y: jax.numpy.ndarray,
            normFunction: callable = jax.numpy.linalg.norm,
            **rhsArgs
    ) -> Tuple[jax.numpy.ndarray, float]:
        dt = self.dt

        def scaled_error_estimate(T: jax.numpy.ndarray, k: int) -> float:
            scale = self.atol + jax.numpy.max(jax.numpy.array([normFunction(y), normFunction(T[k, k])])) * self.rtol
            return normFunction(T[k, k] - T[k, k-1]) / scale

        while True:
            T_matrix = jax.numpy.zeros((self.k_target + 1, self.k_target + 1, y.size), dtype=y.dtype)
            T_matrix = T_matrix.at[0, 0].set(self._midpoint(t, f, y, dt, self.substeps[0], **rhsArgs))

            # Calculate approximation up to k_target - 1
            for k in range(1, self.k_target):
                T_matrix = T_matrix.at[k, 0].set(self._midpoint(t, f, y, dt, self.substeps[k], **rhsArgs))
                for j in range(k):
                    T_matrix = T_matrix.at[k, j+1].set(self._polynomial_extrapolation(T_matrix, k, j))

            err_km2 = scaled_error_estimate(T_matrix, self.k_target - 2)
            H_km2 = self._estimated_step(dt, err_km2, self.k_target - 2)
            W_km2 = self.A_k[self.k_target - 2] / H_km2
            err_km1 = scaled_error_estimate(T_matrix, self.k_target - 1)
            H_km1 = self._estimated_step(dt, err_km1, self.k_target - 1)
            W_km1 = self.A_k[self.k_target - 1] / H_km1

            # Convergence with k - 1?
            if err_km1 <= 1:  # Already accept this approximation
                y_new = T_matrix[self.k_target - 1, self.k_target - 1]
                # Choose new proposed parameters for next step
                if W_km1 < 0.9 * W_km2:
                    self.dt = min(H_km1 * self.A_k[self.k_target] / self.A_k[self.k_target - 1], self.maxStep)
                    self.k_target = self.k_target
                else:
                    self.dt = min(H_km1, self.maxStep)
                    self.k_target = self.k_target - 1
                break
            elif err_km1 > (self.substeps[self.k_target + 1] * self.substeps[self.k_target] / self.substeps[0]
                            / self.substeps[1])**2: # Not expecting convergence in k_target + 1?
                if W_km1 < 0.9 * W_km2:
                    dt = min(H_km1 * self.A_k[self.k_target] / self.A_k[self.k_target - 1], self.maxStep)
                    self.k_target = self.k_target
                else:
                    dt = min(H_km1, self.maxStep)
                    self.k_target = self.k_target - 1
                continue

            # No convergence with k_target - 1 but expected convergence atleast at k_target + 1
            T_matrix = T_matrix.at[self.k_target, 0].set(self._midpoint(t, f, y, dt, self.substeps[self.k_target],
                                                                        **rhsArgs))
            for j in range(self.k_target):
                T_matrix = T_matrix.at[self.k_target, j+1].set(self._polynomial_extrapolation(T_matrix, self.k_target,
                                                                                              j))
            err_k = scaled_error_estimate(T_matrix, self.k_target)
            H_k = self._estimated_step(dt, err_k, self.k_target)
            W_k = self.A_k[self.k_target] / H_k

            # Convergence in k_target?
            if err_k <= 1: # Accept approximation
                y_new = T_matrix[self.k_target, self.k_target]

                # Choose new proposed parameters for next step
                if W_km1 < 0.9 * W_k:
                    self.dt = min(H_km1, self.maxStep)
                    self.k_target = self.k_target - 1
                elif W_k < 0.9 * W_km1:
                    self.dt = min(H_k * self.A_k[self.k_target + 1] / self.A_k[self.k_target], self.maxStep)
                    self.k_target = self.k_target + 1
                else:
                    self.dt = min(H_k, self.maxStep)
                    self.k_target = self.k_target
                break
            elif err_k > (self.substeps[self.k_target + 1] / self.substeps[0])**2:
                # Choose proposed parameters for retrying this step
                if W_km1 < 0.9 * W_k:
                    dt = min(H_km1, self.maxStep)
                    self.k_target = self.k_target - 1
                elif W_k < 0.9 * W_km1:
                    dt = min(H_k * self.A_k[self.k_target + 1] / self.A_k[self.k_target], self.maxStep)
                    self.k_target = self.k_target + 1
                else:
                    dt = min(H_k, self.maxStep)
                    self.k_target = self.k_target
                continue

            # No convergence in k_target, but maybe with k_target + 1
            T_matrix = T_matrix.at[self.k_target + 1, 0].set(self._midpoint(t, f, y, dt, self.substeps[self.k_target+1],
                                                                            **rhsArgs))
            for j in range(self.k_target + 1):
                T_matrix = T_matrix.at[self.k_target+1, j+1].set(self._polynomial_extrapolation(T_matrix,
                                                                                                self.k_target + 1, j))

            err_kp1 = scaled_error_estimate(T_matrix, self.k_target + 1)
            H_kp1 = self._estimated_step(dt, err_kp1, self.k_target + 1)
            W_kp1 = self.A_k[self.k_target + 1] / H_kp1

            if err_kp1 <= 1:
                y_new = T_matrix[self.k_target + 1, self.k_target + 1]

                if W_km1 < 0.9 * W_k:
                    self.k_target = self.k_target - 1
                if W_kp1 < 0.9 * W_k:
                    self.k_target = self.k_target + 1
                else:
                    self.k_target = self.k_target
                break
            else:
                dt = min(H_kp1, self.maxStep)
                continue

        return y_new, dt
