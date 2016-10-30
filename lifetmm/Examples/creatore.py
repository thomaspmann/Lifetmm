"""
Script to recreate the plots in the paper

    'Quantum theory of spontaneous emission in multilayer dielectric structures'
    by Celestino Creatore and Lucio Claudio Andreani.
"""

import matplotlib.pyplot as plt
from lifetmm.Methods.SpontaneousEmissionRate import *
SAVE = False


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

    # Feedback to user the structure being simulated
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.spe_structure_radiative()
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.z_to_lambda(z)

    # Plot spontaneous emission rates
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(15, 7))
    ax1.plot(z, spe['TE_total'], label='TE')
    ax1.plot(z, spe['TM_p_total'], label='TM')
    ax1.plot(z, spe['TE_total']+spe['TM_p_total'], label='TE + TM')

    ax2.plot(z, spe['TE_lower_full'] + spe['TM_p_lower_full'], label='Fully radiative lower outgoing')
    ax2.plot(z, spe['TE_lower_partial'] + spe['TM_p_lower_partial'], label='Partially radiative lower outgoing')
    ax2.plot(z, spe['TE_upper'] + spe['TM_p_upper'], label='Fully radiative upper outgoing')

    ax3.plot(z, spe['TM_s_total'], label='TM')

    ax4.plot(z, spe['TM_s_lower_full'], label='Fully radiative lower outgoing')
    ax4.plot(z, spe['TM_s_lower_partial'], label='Partially radiative lower outgoing')
    ax4.plot(z, spe['TM_s_upper'], label='Fully radiative upper outgoing')

    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.z_to_lambda(z), color='k', lw=2)
        ax2.axvline(st.z_to_lambda(z), color='k', lw=2)
        ax3.axvline(st.z_to_lambda(z), color='k', lw=2)
        ax4.axvline(st.z_to_lambda(z), color='k', lw=2)

    ax1.set_ylim(0, 4)
    ax3.set_ylim(0, 6)
    ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax3.set_ylabel('$\Gamma /\Gamma_0$')
    ax3.set_xlabel('z/$\lambda$')
    ax4.set_xlabel('z/$\lambda$')
    size = 12
    ax1.legend(title='Horizontal Dipoles', prop={'size': size})
    ax2.legend(title='Horizontal Dipoles', prop={'size': size})
    ax3.legend(title='Vertical Dipoles', prop={'size': size})
    ax4.legend(title='Vertical Dipoles', prop={'size': size})
    fig.tight_layout()
    if SAVE:
        plt.savefig('../Images/spe_vs_n.png', dpi=300)
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
    result = st.spe_structure_radiative()
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.z_to_lambda(z)

    # Plot spontaneous emission rates
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe['TE_total'], label='TE')
    ax1.plot(z, spe['TM_p_total'], label='TM')
    ax1.plot(z, spe['TE_total']+spe['TM_p_total'], 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe['TM_s_total'], label='TM')

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
    if SAVE:
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
    result = st.spe_structure_radiative()
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.z_to_lambda(z)

    # Plot spontaneous emission rates
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe['TE_total'], label='TE')
    ax1.plot(z, spe['TM_p_total'], label='TM')
    ax1.plot(z, spe['TE_total']+spe['TM_p_total'], 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe['TM_s_total'], label='TM')

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
    fig3()
    fig6()
    # figx()
