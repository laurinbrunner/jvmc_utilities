import jax.numpy as jnp
import jVMC
from jvmc_utilities import *


def test_higher_order_M_T_inv():
    povm = jVMC.operator.POVM({"dim": "1D", "L": 2})

    _M_2 = jnp.array([[jnp.kron(povm.M[i], povm.M[j]) for j in range(4)] for i in range(4)]).reshape(4**2, 2**2, 2**2)
    _T_2 = jnp.kron(povm.T_inv, povm.T_inv)

    M_2, T_2 = higher_order_M_T_inv(2, povm.M, povm.T_inv)

    assert(jnp.allclose(_M_2, M_2))
    assert(jnp.allclose(_T_2, T_2))

    _M_3 = jnp.array([[[jnp.kron(jnp.kron(povm.M[i], povm.M[j]), povm.M[k]) for k in range(4)] for j in range(4)]
                      for i in range(4)]).reshape(4**3, 2**3, 2**3)
    _T_3 = jnp.kron(jnp.kron(povm.T_inv, povm.T_inv), povm.T_inv)

    M_3, T_3 = higher_order_M_T_inv(3, povm.M, povm.T_inv)

    assert(jnp.allclose(_M_3, M_3))
    assert(jnp.allclose(_T_3, T_3))


def test_initalisation_operators():
    povm = jVMC.operator.POVM({"dim":  "1D", "L": 2})

    assert("upup_dis" not in povm.operators.keys())
    assert("updown_dis" not in povm.operators.keys())
    assert("downup_dis" not in povm.operators.keys())
    assert("downdown_dis" not in povm.operators.keys())
    assert("up_dis" not in povm.operators.keys())
    assert("down_dis" not in povm.operators.keys())

    initialisation_operators(povm)

    assert("upup_dis" in povm.operators.keys())
    assert("updown_dis" in povm.operators.keys())
    assert("downup_dis" in povm.operators.keys())
    assert("downdown_dis" in povm.operators.keys())
    assert("up_dis" in povm.operators.keys())
    assert("down_dis" in povm.operators.keys())

    try:
        initialisation_operators(povm)
    except ValueError as exc:
        assert False, f"'initialisation_operators' raised an exception {exc}"


def test_aqi_model_operators():
    povm = jVMC.operator.POVM({"dim": "1D", "L": 2})

    assert("spin_flip_uni" not in povm.operators.keys())
    assert("spin_flip_dis" not in povm.operators.keys())

    aqi_model_operators(povm)

    assert("spin_flip_uni" in povm.operators.keys())
    assert("spin_flip_dis" in povm.operators.keys())

    try:
        aqi_model_operators(povm)
    except ValueError as exc:
        assert False, f"'initialisation_operators' raised an exception {exc}"
