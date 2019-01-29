import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from lifetmm.SPE import SPE


def plot_rates(d, n, n_sub, title):
    """ SE rate for Silicon to air semi-infinite half spaces."""
    # """Plot leaky and guided SE rates and then sum for randomly orientated dipole."""

    lam0 = 1540

    # Create structure
    st = SPE()
    st.add_layer(.5 * lam0, n_sub)
    st.add_layer(d, n)
    st.add_layer(.5 * lam0, 1)
    st.set_vacuum_wavelength(lam0)
    st.info()

    # Calculate
    res = st.calc_spe_structure(th_pow=11)
    z = res['z']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # ------- Plots -------
    # Plot data
    fig, ax1 = plt.subplots()
    if st.supports_guiding():
        ax1.plot(z, res['leaky']['avg'] + res['guided']['avg'], label='Avg')
        ax1.plot(z, res['leaky']['parallel'] + res['guided']['parallel'], '--', label=r'$\parallel$')
        ax1.plot(z, res['leaky']['perpendicular'] + res['guided']['perpendicular'], '-.', label=r'$\bot$')
        fp = np.mean(res['leaky']['avg'] + res['guided']['avg'])
    else:
        ax1.plot(z, res['leaky']['avg'], label='Avg')
        ax1.plot(z, res['leaky']['parallel'], '--', label=r'$\parallel$')
        ax1.plot(z, res['leaky']['perpendicular'], '-.', label=r'$\bot$')
        fp = np.mean(res['leaky']['avg'])
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z [$\lambda$]')
    ax1.legend(fontsize='small')
    ax1.set_ylim(0, ax1.get_ylim()[1])
    bounds = ax1.get_ylim()

    print('Purcell factor = {}'.format(fp))

    # Draw rectangles for the refractive index
    ax2 = ax1.twinx()
    for z0, dz, n in zip(st.d_cumulative, st.d_list, st.n_list):
        z0 = st.calc_z_to_lambda(z0)
        dz = st.calc_z_to_lambda(dz, center=False)
        rect = Rectangle((z0 - dz, 0), dz, n.real, facecolor='c', alpha=0.15)
        ax2.add_patch(rect)  # Note: add to ax1 so that zorder has effect
    ax2.set_ylabel('n')
    ax2.set_ylim(bounds)

    ax1.set_zorder(ax2.get_zorder() + 1)  # put ax1 in front of ax2
    ax1.patch.set_visible(False)  # hide ax1'canvas'

    for zb in st.get_layer_boundaries()[:-1]:
        zb = st.calc_z_to_lambda(zb)
        ax1.axvline(x=zb, color='k', lw=2)
    ax1.set_xlim([min(z), max(z)])
    if SAVE:
        plt.savefig('../Images/' + title)
    # plt.show()


# Update plot parameters for publication
def update():
    # Set figure size
    WIDTH = 412.564  # the number (in pt) latex spits out when typing: \the\linewidth (paper 246, thesis 412.56)
    FACTOR = 0.75  # the fraction of the width you'd like the figure to occupy
    fig_width_pt = WIDTH * FACTOR

    inches_per_pt = 1.0 / 72.27
    golden_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

    fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio  # figure height in inches
    fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list

    # Update rcParams for figure size
    params = {
        'font.size': 11.0,
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'cm',
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'figure.figsize': fig_dims,
    }
    plt.rcParams.update(params)


if __name__ == "__main__":
    SAVE = True
    update()

    # plot_rates(437, 1.563, 1.453, 'OH/SPE_Infrasil_301')
    # plot_rates(531, 1.546, 1.474, 'OH/SPE_Borofloat_33')
    # plot_rates(358, 1.587, 1.454, 'OH/SPE_Spectrosil_2000')
    # plot_rates(371, 1.579, 1.452, 'OH/SPE_Corning_7980')
    # plot_rates(297, 1.611, 1.454, 'OH/SPE_B36')
    plot_rates(297, 1.611, 3.4757, 'OH/SPE_B36_Si')
