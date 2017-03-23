"""
Script to recreate the plots in the paper

    'Quantum theory of spontaneous emission in multilayer dielectric structures'
    by Celestino Creatore and Lucio Claudio Andreani.
"""
import matplotlib.pyplot as plt
import numpy as np

from lifetmm.SpontaneousEmissionRate import LifetimeTmm


def fig3():
    """
    Silicon to air semi-infinite half spaces.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(lam0, si)
    st.add_layer(lam0, air)
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.calc_spe_structure_leaky()
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # Plot spontaneous emission rates
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(15, 7))
    ax1.plot(z, spe['TE'], label='TE')
    ax1.plot(z, spe['TM_p'], label='TM')
    ax1.plot(z, spe['TE'] + spe['TM_p'], label='TE + TM')

    ax2.plot(z, spe['TE_lower_full'] + spe['TM_p_lower_full'], label='Fully radiative lower outgoing')
    ax2.plot(z, spe['TE_lower_partial'] + spe['TM_p_lower_partial'], label='Partially radiative lower outgoing')
    ax2.plot(z, spe['TE_upper'] + spe['TM_p_upper'], label='Fully radiative upper outgoing')

    ax3.plot(z, spe['TM_s'], label='TM')

    ax4.plot(z, spe['TM_s_lower_full'], label='Fully radiative lower outgoing')
    ax4.plot(z, spe['TM_s_lower_partial'], label='Partially radiative lower outgoing')
    ax4.plot(z, spe['TM_s_upper'], label='Fully radiative upper outgoing')

    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax2.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax3.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax4.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')

    ax1.set_ylim(0, 4)
    ax3.set_ylim(0, 6)
    ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax3.set_ylabel('$\Gamma /\Gamma_0$')
    ax3.set_xlabel('z/$\lambda$')
    ax4.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', fontsize='small')
    ax2.legend(title='Horizontal Dipoles', fontsize='small')
    ax3.legend(title='Vertical Dipoles', fontsize='small')
    ax4.legend(title='Vertical Dipoles', fontsize='small')
    fig.tight_layout()
    if SAVE:
        plt.savefig('../Images/creatore_fig3')
    plt.show()


def fig4():
    """
    Silicon to air semi-infinite half spaces.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(lam0, si)
    st.add_layer(lam0, air)
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.calc_spe_structure_leaky(th_pow=10)
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # Plot spontaneous emission rates
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(15, 5))
    ax1.plot(z, (spe['TM_p_lower'] + spe['TE_lower']) / (spe['TE'] + spe['TM_p']), label='Lower')
    ax1.plot(z, (spe['TM_p_upper'] + spe['TE_upper']) / (spe['TE'] + spe['TM_p']), label='Upper')

    ax2.plot(z, (spe['TM_s_lower']) / spe['TM_s'], label='Lower')
    ax2.plot(z, (spe['TM_s_upper']) / spe['TM_s'], label='Upper')

    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax2.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')

    ax1.set_ylim(0, 1.1)

    ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('z/$\lambda$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', fontsize='small')
    ax2.legend(title='Vertical Dipoles', fontsize='small')

    fig.tight_layout()
    if SAVE:
        plt.savefig('../Images/creatore_fig4')
    plt.show()


def fig5():
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(1.5 * lam0, air)
    st.add_layer(lam0, si)
    st.add_layer(1.5 * lam0, air)
    st.info()

    result = st.calc_spe_structure_guided()
    z = result['z']
    spe = result['spe']
    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')

    ax1.plot(z, spe['TE'], label='TE')
    ax1.plot(z, spe['TM_p'], label='TM')
    ax1.plot(z, spe['TE'] + spe['TM_p'], label='TE + TM')
    ax2.plot(z, spe['TM_s'], label='TM')
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax2.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
    # ax1.set_ylim(0, 4)
    # ax2.set_ylim(0, 6)
    ax1.set_title('Spontaneous Emission Rate. Core n=3.48, Cladding n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    size = 12
    ax1.legend(title='Horizontal Dipoles', prop={'size': size})
    ax2.legend(title='Vertical Dipoles', prop={'size': size})

    fig.tight_layout()
    if SAVE:
        plt.savefig('../Images/creatore_fig5')
    plt.show()


def fig6():
    """
    Silicon layer bounded by two semi infinite air claddings.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(2.5 * lam0, air)
    st.add_layer(lam0, si)
    st.add_layer(2.5 * lam0, air)
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.calc_spe_structure_leaky()
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # Plot spontaneous emission rates
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe['TE'], label='TE')
    ax1.plot(z, spe['TM_p'], label='TM')
    ax1.plot(z, spe['TE'] + spe['TM_p'], 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe['TM_s'], label='TM')

    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax2.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')

    ax1.set_ylim(0, 1.4)
    ax2.set_ylim(0, 1.4)
    ax1.set_title('Spontaneous Emission Rate. Silicon (n=3.48) with air cladding (n=1.)')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', loc='lower right', fontsize='medium')
    ax2.legend(title='Vertical Dipoles', loc='lower right', fontsize='medium')
    if SAVE:
        plt.savefig('../Images/creatore_fig6')
    plt.show()


def fig7():
    """
    Silicon layer bounded by two semi infinite air claddings.
    """
    d_list = np.arange(start=11, stop=2511, step=5)
    te_guided = []
    tm_guided_p = []
    te_leaky = []
    tm_leaky_p = []
    tm_guided_s = []
    tm_leaky_s = []

    k0 = 2 * np.pi / lam0
    for d in d_list:

        # Create structure
        st = LifetimeTmm()
        st.set_vacuum_wavelength(lam0)
        st.add_layer(0, air)
        st.add_layer(d, si)
        st.add_layer(0, air)
        st.info()

        # Calculate spontaneous emission of layer
        result = st.calc_spe_structure(th_pow=9)
        z = result['z']
        iloc = int((len(z) - 1) / 2)
        leaky = result['leaky']
        try:
            guided = result['guided']
            te_guided.append(guided['TE'][iloc])
            tm_guided_p.append(guided['TM_p'][iloc])
            tm_guided_s.append(guided['TM_s'][iloc])
        except KeyError:
            te_guided.append(0)
            tm_guided_p.append(0)
            tm_guided_s.append(0)

        te_leaky.append(leaky['TE'][iloc])
        tm_leaky_p.append(leaky['TM_p'][iloc])
        tm_leaky_s.append(leaky['TM_s'][iloc])

    # Convert lists to arrays
    d_list = np.array(d_list)
    te_guided = np.array(te_guided)
    tm_guided_p = np.array(tm_guided_p)
    tm_guided_s = np.array(tm_guided_s)
    te_leaky = np.array(te_leaky)
    tm_leaky_p = np.array(tm_leaky_p)
    tm_leaky_s = np.array(tm_leaky_s)

    # Plot spontaneous emission rates
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(d_list * k0, te_guided, label='TE guided')
    ax1.plot(d_list * k0, tm_guided_p, label='TM guided')
    ax1.plot(d_list * k0, te_leaky, 'k', label='TE leaky')
    ax1.plot(d_list * k0, tm_leaky_p, 'k', label='TM leaky')
    total = tm_leaky_p + te_leaky + tm_guided_p + te_guided
    ax1.plot(d_list * k0, total, 'k', label='total')

    ax2 = fig.add_subplot(212)
    ax2.plot(d_list * k0, tm_guided_s, 'k', label='TM guided')
    ax2.plot(d_list * k0, tm_leaky_s * 20, 'k', label='TM leaky')

    # ax1.set_ylim(0, 1.4)
    # ax2.set_ylim(0, 1.4)
    # ax1.set_title('Spontaneous Emission Rate. Silicon (n=3.48) with air cladding (n=1.)')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    # ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', loc='lower right', fontsize='medium')
    ax2.legend(title='Vertical Dipoles', loc='lower right', fontsize='medium')
    if SAVE:
        plt.savefig('../Images/creatore_fig6')
    plt.show()


def fig8():
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(1.5 * lam0, sio2)
    st.add_layer(lam0, si)
    st.add_layer(1.5 * lam0, air)
    st.info()

    # Do Simulation
    result = st.calc_spe_structure_guided()
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')

    ax1.plot(z, spe['TE'], label='TE')
    ax1.plot(z, spe['TM_p'], label='TM')
    ax1.plot(z, spe['TE'] + spe['TM_p'], label='TE + TM')
    ax2.plot(z, spe['TM_s'], label='TM')
    for z in st.get_layer_boundaries()[:-1]:
        z = st.calc_z_to_lambda(z)
        ax1.axvline(x=z, color='k', lw=1, ls='--')
        ax2.axvline(x=z, color='k', lw=1, ls='--')
    ax1.set_ylim(0, 4)
    ax2.set_ylim(0, 5)
    ax1.set_title('The spatial dependence of the normalized spontaneous emission rate \n'
                  'into guided modes for asymmetric Silicon waveguide (SiO2/Si/air).')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', fontsize='small')
    ax2.legend(title='Vertical Dipoles', fontsize='small')

    fig.tight_layout()
    if SAVE:
        plt.savefig('../Images/creatore_fig8')
    plt.show()


def fig9():
    """
    Silicon layer bounded by two semi infinite air claddings.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(2.5 * lam0, sio2)
    st.add_layer(lam0, si)
    st.add_layer(2.5 * lam0, air)
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.calc_spe_structure_leaky()
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # Plot spontaneous emission rates
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe['TE'], label='TE')
    ax1.plot(z, spe['TM_p'], label='TM')
    ax1.plot(z, spe['TE'] + spe['TM_p'], 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe['TM_s'], label='TM')

    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')
        ax2.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')

    ax1.set_ylim(0, 2)
    ax2.set_ylim(0, 3)
    ax1.set_title('The spatial dependence of the normalized spontaneous emission rate \n'
                  'into leaky modes for asymmetric Silicon waveguide (SiO2/Si/air).')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles', loc='upper right', fontsize='small')
    ax2.legend(title='Vertical Dipoles', loc='upper right', fontsize='small')
    if SAVE:
        plt.savefig('../Images/creatore_fig9')
    plt.show()


def fig13a():
    """
    Silicon layer bounded by two semi infinite air claddings.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(1e3, si)
    st.add_layer(1900, sio2)
    st.add_layer(100, si)
    st.add_layer(20, sio2)
    st.add_layer(100, si)
    st.add_layer(1e3, air)
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.calc_spe_structure_leaky(th_pow=12)
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # Plot spontaneous emission rates
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(z, spe['avg'])

    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')

    ax1.axhline(si, color='k')
    ax1.axhline(sio2, color='k')
    ax1.axhline(air, color='k')
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(0, 4.5)
    # ax1.set_title('The spatial dependence of the normalized spontaneous emission rate \n'
    #               'into leaky modes for asymmetric Silicon waveguide (SiO2/Si/air).')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('z/$\lambda$')

    if SAVE:
        plt.savefig('../Images/creatore_fig13a')
    plt.show()


def fig13b():
    """
    Silicon layer bounded by two semi infinite air claddings.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    # st.add_layer(1e3, si)
    st.add_layer(1900, sio2)
    st.add_layer(100, si)
    st.add_layer(20, sio2)
    st.add_layer(100, si)
    st.add_layer(1e3, air)
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.calc_spe_structure_guided()
    z = result['z']
    spe = result['spe']

    # Convert z into z/lam0 and center
    z = st.calc_z_to_lambda(z)

    # Plot spontaneous emission rates
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(z, spe['TE'], label=r'$\Gamma^{\mathrm{TE}}$')
    ax1.plot(z, spe['TM_p'], label=r'$\Gamma^{\mathrm{TM}}_∥$')
    ax1.plot(z, spe['TM_s'], label=r'$\Gamma^{\mathrm{TM}}_⊥$')
    ax1.plot(z, spe['avg'], label=r'total')
    ax1.legend()

    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(st.calc_z_to_lambda(z), color='k', lw=1, ls='--')

    ax1.set_ylim(0, 16)
    ax1.set_xlim(0, 0.5)
    # ax1.set_title('The spatial dependence of the normalized spontaneous emission rate \n'
    #               'into leaky modes for asymmetric Silicon waveguide (SiO2/Si/air).')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('z/$\lambda$')

    if SAVE:
        plt.savefig('../Images/creatore_fig13b')
    plt.show()

if __name__ == "__main__":
    SAVE = False

    # Set vacuum wavelength
    lam0 = 1550

    # Material refractive index at lam0
    sio2 = 1.45
    si = 3.48
    air = 1

    # fig3()
    # fig4()
    # fig5()
    # fig6()
    # fig7()
    # fig8()
    # fig9()
    # fig13a()
    fig13b()
