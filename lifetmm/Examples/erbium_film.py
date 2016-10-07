"""
Script to recreate the plots in the paper

    'Quantum theory of spontaneous emission in multilayer dielectric structures'
    by Celestino Creatore and Lucio Claudio Andreani.

"""

import matplotlib.pyplot as plt
from lifetmm.Methods.SpontaneousEmissionRate import *


def medium():
    """ Silicon to air semi-infinite half spaces.
    """
    # Create structure
    st = LifetimeTmm()
    st.add_layer(1550, 3)
    st.add_layer(1550, 1.55)

    # Set vacuum wavelength
    lam0 = 1550
    st.set_wavelength(lam0)

    # Calculate spontaneous emission over whole structure
    result = st.spe_structure()
    z = result['z']
    spe_rates = result['spe_rates']
    spe_TE = spe_rates['spe_TE']
    spe_TM_p = spe_rates['spe_TM_p']
    spe_TM_s = spe_rates['spe_TM_s']

    # Plot spontaneous emission rates
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe_TE, label='TE')
    ax1.plot(z, spe_TM_p, label='TM')
    ax1.plot(z, spe_TE+spe_TM_p, 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe_TM_s, label='TM')

    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(z, color='r', lw=2)
        ax2.axvline(z, color='r', lw=2)

    ax1.set_title('Spontaneous Emission Rate. LHS n=3, RHS n=1.55.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles')
    ax2.legend(title='Vertical Dipoles')
    plt.savefig('../Images/SPE_er_n=3.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    medium()
