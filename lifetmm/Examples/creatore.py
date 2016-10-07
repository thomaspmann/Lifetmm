"""
Script to recreate the plots in the paper

    'Quantum theory of spontaneous emission in multilayer dielectric structures'
    by Celestino Creatore and Lucio Claudio Andreani.

"""

import matplotlib.pyplot as plt
from lifetmm.Methods.SpontaneousEmissionRate import *


def fig3():
    """ Silicon to air semi-infinite half spaces.
    """
    # Create structure
    st = LifetimeTmm()
    st.add_layer(1550, 3.48)
    st.add_layer(1550, 1)

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

    # Convert z into z/lam0 and center
    z = st.z_to_lambda(z)

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
        ax1.axvline(st.z_to_lambda(z), color='r', lw=2)
        ax2.axvline(st.z_to_lambda(z), color='r', lw=2)

    ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles')
    ax2.legend(title='Vertical Dipoles')
    plt.savefig('../Images/SPE_n_3.38_to_1.png', dpi=300)
    plt.show()


def fig3p5():
    """ Silicon to air semi-infinite half spaces.
    """
    # Create structure
    st = LifetimeTmm()
    st.add_layer(1550, 3.48)
    st.add_layer(1550, 1)

    # Set vacuum wavelength
    lam0 = 1550
    st.set_wavelength(lam0)

    # Calculate spontaneous emission over whole structure
    result = st.spe_structure()
    z = result['z']
    spe_rates = result['spe_rates']
    spe_TE_lower = spe_rates['spe_TE_lower']
    spe_TM_p_lower = spe_rates['spe_TM_p']
    spe_TM_s = spe_rates['spe_TM_s']

    # Convert z into z/lam0 and center
    z = st.z_to_lambda(z)

    # Plot spontaneous emission rates
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe_TE_lower, label='TE')
    ax1.plot(z, spe_TM_p_lower, label='TM')
    ax1.plot(z, spe_TE_lower+spe_TM_p_lower, 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe_TM_s, label='TM')

    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.z_to_lambda(z), color='r', lw=2)
        ax2.axvline(st.z_to_lambda(z), color='r', lw=2)

    ax1.set_ylim(0, 4)
    ax2.set_ylim(0, 6)
    ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles')
    ax2.legend(title='Vertical Dipoles')
    # plt.savefig('../Images/SPE_n_3.38_to_1.png', dpi=300)
    plt.show()


def fig6():
    """ Silicon layer bounded by two semi infinite air claddings.
    """
    # Create structure
    st = LifetimeTmm()

    # Set vacuum wavelength
    lam0 = 1550
    st.set_wavelength(lam0)

    # Add layers
    st.add_layer(2.5*lam0, 1)
    st.add_layer(lam0, 3.48)
    st.add_layer(2.5*lam0, 1)

    # Calculate spontaneous emission over whole structure
    result = st.spe_structure()
    z = result['z']
    spe_rates = result['spe_rates']
    spe_TE = spe_rates['spe_TE']
    spe_TM_p = spe_rates['spe_TM_p']
    spe_TM_s = spe_rates['spe_TM_s']

    # Convert z into z/lam0 and center
    z = st.z_to_lambda(z)

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
        ax1.axvline(st.z_to_lambda(z), color='r', lw=2)
        ax2.axvline(st.z_to_lambda(z), color='r', lw=2)

    ax1.set_ylim(0, 1.4)
    ax2.set_ylim(0, 1.4)
    ax1.set_title('Spontaneous Emission Rate. Silicon (n=3.48) with air cladding (n=1.)')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', loc='lower right', fontsize='medium')
    ax2.legend(title='Vertical Dipoles', loc='lower right', fontsize='medium')
    plt.savefig('../Images/SPE_silicon_layer_air_cladding.png', dpi=300)
    plt.show()


def figx():
    """ Air layer bounded by two semi infinite silicon claddings.
    """
    # Create structure
    st = LifetimeTmm()

    # Set vacuum wavelength
    lam0 = 1550
    st.set_wavelength(lam0)

    # Add layers
    st.add_layer(lam0, 3.48)
    st.add_layer(lam0, 1)
    st.add_layer(lam0, 3.48)

    # Calculate spontaneous emission over whole structure
    result = st.spe_structure()
    z = result['z']
    spe_rates = result['spe_rates']
    spe_TE = spe_rates['spe_TE']
    spe_TM_p = spe_rates['spe_TM_p']
    spe_TM_s = spe_rates['spe_TM_s']

    # Convert z into z/lam0 and center
    z = st.z_to_lambda(z)

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
        ax1.axvline(st.z_to_lambda(z), color='r', lw=2)
        ax2.axvline(st.z_to_lambda(z), color='r', lw=2)

    ax1.set_title('Spontaneous Emission Rate. Air (n=1) with silicon cladding (n=3.48).')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', loc='lower right', fontsize='medium')
    ax2.legend(title='Vertical Dipoles', loc='lower right', fontsize='medium')
    plt.savefig('../Images/SPE_air_layer_silicon_cladding.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    fig3p5()
    # fig6()
    # figx()
