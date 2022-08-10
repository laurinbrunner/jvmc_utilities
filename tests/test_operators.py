import jax.numpy as jnp
import jVMC
from jvmc_utilities import higher_order_M_T_inv
import jvmc_utilities


def test_higher_order_M_T_inv():
    povm = jVMC.operator.POVM({"dim": "1D", "L": 2})

    _M_2 = jnp.array([[jnp.kron(povm.M[i], povm.M[j]) for j in range(4)] for i in range(4)]).reshape(4**2, 2**2, 2**2)
    _T_2 = jnp.kron(povm.T_inv, povm.T_inv)

    M_2, T_2 = higher_order_M_T_inv(2, povm.M, povm.T_inv)

    assert(jnp.allclose(_M_2, M_2))
    assert(jnp.allclose(_T_2, T_2))
