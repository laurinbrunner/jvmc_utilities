import numpy as np
import jax.numpy as jnp
from jVMC.util.symmetries import LatticeSymmetry


def get_orbit_aqi_translation(L):
    """
    Gives translational symmetry operations that translate the computational system by 2*n steps.

    :param L: Physical system size
    :return: jVMC.util.symmetries.LatticeSymmetry object
    """
    translations = np.array([np.eye(2*L)] * L)
    for idx, t, in enumerate(translations):
        translations[idx] = np.roll(t, 2*idx, axis=1)

    translations = jnp.array(translations)

    return LatticeSymmetry(translations.reshape(-1, 2*L, 2*L).astype(np.int32), jnp.ones(L))
