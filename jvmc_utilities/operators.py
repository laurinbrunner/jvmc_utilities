import jax.numpy as jnp
import jVMC.operator as jvmcop
import itertools
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

    sigmas = jvmcop.get_paulis()

    # The following are spin-1/2 ladder operators for the \sigma_z, \sigma_x and \sigma_y basis respectively
    sz_plus = (sigmas[0] + 1j * sigmas[1]) / 2
    sz_minus = (sigmas[0] - 1j * sigmas[1]) / 2

    spin_flip = jnp.kron(sz_plus, sz_minus)
    spin_flip_Z = jnp.kron(jnp.kron(sz_plus, sz_minus), sigmas[2])
    inv_spin_flip = jnp.kron(sz_minus, sz_plus)
    inv_spin_flip_Z = jnp.kron(jnp.kron(sz_minus, sz_plus), sigmas[2])

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
