import matplotlib.pyplot as plt
import jax.numpy as jnp


def plotting(results, times=None):
    """
    Plots the particle density and magnetization as well as Sx, Sy and Sz for all sites.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    if times is None:
        times = jnp.arange(results["N"].shape[0])
    L = results["Sz_i"].shape[1]

    ax1.plot(times, (results["N"][:, 0] + results["N"][:, 1]), label=r"$\langle N \rangle$")
    ax1.plot(times, (results["N"][:, 0] - results["N"][:, 1]), label=r"$\langle M \rangle$")

    for i in range(L):
        str_i = str(i)
        ax2.plot(times, results["Sx_i"][:, i], label=r"$\langle \sigma_{x," + str_i + r"} \rangle$")

        ax3.plot(times, results["Sy_i"][:, i], label=r"$\langle \sigma_{y," + str_i + r"} \rangle$")

        ax4.plot(times, results["Sz_i"][:, i], label=r"$\langle \sigma_{z," + str_i + r"} \rangle$")

    ax1.legend()
    ax1.grid()
    ax2.legend()
    ax2.grid()
    ax3.legend()
    ax3.grid()
    ax4.legend()
    ax4.grid()
    ax2.yaxis.tick_right()
    ax4.yaxis.tick_right()

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0)

    return fig, ((ax1, ax2), (ax3, ax4))

