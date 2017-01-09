"""
Thin film calculations.
"""

import matplotlib.pyplot as plt
import numpy as np

from lifetmm.Methods.SpontaneousEmissionRate import LifetimeTmm


def t2():
    """
    T2 EDTS layer next to air.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(2 * lam0, sio2)
    st.add_layer(d_etds, edts)
    st.add_layer(2 * lam0, air)
    st.info()

    # Calculate spontaneous emission for leaky and guided modes
    result = st.calc_spe_structure(th_pow=9)
    z = result['z']
    z = st.calc_z_to_lambda(z)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(z, result['leaky']['avg'], label='leaky')
    try:
        ax2.plot(z, result['guided']['avg'], label='guided')
    except KeyError:
        pass

    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        z = st.calc_z_to_lambda(z)
        ax1.axvline(z, color='k', lw=1, ls='--')
        ax2.axvline(z, color='k', lw=1, ls='--')
    # ax1.set_title('Spontaneous emission rate at boundary for semi-infinite media. LHS n=1.57.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('Position z ($\lambda$/2$\pi$)')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    if SAVE:
        plt.savefig('../Images/t2.png', dpi=300)
    plt.show()


def t2_leaky():
    """
    T2 EDTS layer next to air.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(2 * lam0, sio2)
    st.add_layer(d_etds, edts)
    st.add_layer(2 * lam0, air)
    st.info()

    # Calculate spontaneous emission for leaky and guided modes
    result = st.calc_spe_structure_leaky(th_pow=9)
    z = result['z']
    z = st.calc_z_to_lambda(z)
    spe = result['spe']

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

    # ax1.set_ylim(0, 4)
    # ax3.set_ylim(0, 6)
    # ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
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
        plt.savefig('../Images/t2_leaky.png', dpi=300)
    plt.show()


def t2_fig4():
    """
    Silicon to air semi-infinite half spaces.
    """
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(2 * lam0, sio2)
    st.add_layer(d_etds, edts)
    st.add_layer(2 * lam0, air)
    st.info()

    # Calculate spontaneous emission over whole structure
    result = st.calc_spe_structure_leaky(th_pow=9)
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

    # ax1.set_ylim(0, 1.1)

    # ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('z/$\lambda$')
    ax2.set_xlabel('z/$\lambda$')
    ax1.legend(title='Horizontal Dipoles')
    ax2.legend(title='Vertical Dipoles')

    fig.tight_layout()
    if SAVE:
        plt.savefig('../Images/t2_fig4.png', dpi=300)
    plt.show()


def t2_guided():
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(2 * lam0, sio2)
    st.add_layer(d_etds, edts)
    st.add_layer(2 * lam0, air)
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
        z = st.calc_z_to_lambda(z)
        ax1.axvline(x=z, color='k', lw=2, zorder=-1)
        ax2.axvline(x=z, color='k', lw=2, zorder=-1)
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
        plt.savefig('../Images/t2_guided.png', dpi=300)
    plt.show()


def t2_spe_vs_n():
    # n_list = np.append(np.linspace(1, 1.45, num=25), np.linspace(1.45, 1.55, num=50))
    # n_list = np.append(n_list, np.linspace(1.55, 2, num=25))
    n_list = [air, water, 1.3675, 1.47]
    spe_list = []
    leaky_list = []
    guided_list = []
    for n in n_list:
        print('Evaluating n={:g}'.format(n))

        # Create structure
        st = LifetimeTmm()
        st.set_vacuum_wavelength(lam0)
        st.add_layer(0, sio2)
        st.add_layer(d_etds, edts)
        st.add_layer(0, n)
        st.info()

        # Calculate spontaneous emission of layer 0 (1st)
        result = st.calc_spe_structure(th_pow=15)
        leaky = result['leaky']['avg']
        try:
            guided = result['guided']['avg']
        except KeyError:
            guided = 0

        # Average over layer
        leaky = np.mean(leaky)
        guided = np.mean(guided)
        # Append to list
        leaky_list.append(leaky)
        guided_list.append(guided)
        spe_list.append(leaky + guided)

    # Convert lists to arrays
    n_list = np.array(n_list)
    leaky_list = np.array(leaky_list)
    guided_list = np.array(guided_list)
    spe_list = np.array(spe_list)
    print(n_list, spe_list)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='none')
    ax1.plot(n_list, spe_list, '.-', label='leaky + guided')
    ax2.plot(n_list, leaky_list, '.-', label='leaky')
    ax3.plot(n_list, guided_list, '.-', label='guided')
    ax3.set_xlim(1, 2)
    ax1.set_title('Average Spontaneous Emission Rate for Random Orientated Dipole in T2.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('n')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout()

    if SAVE:
        plt.savefig('../Images/t2_vs_n.png', dpi=300)
        np.savez('../Data/t2_vs_n', n=n_list, spe=spe_list, guided=guided_list, leaky=leaky_list)
    plt.show()


def load_data():
    data = np.load('./lifetmm/Data/spe_vs_n.npz')
    print(data._files)


def purcell_factor():
    """
    Photonic chip next to two mediums.
    Leaky and guided separate plots.
    Evaluate purcell factor for randomly orientated dipole averaged over film thickness.
    """
    # Medium 1
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(1.5 * lam0, sio2)
    st.add_layer(d_etds, edts)
    st.add_layer(1.5 * lam0, air)
    st.info()

    # Calculate spontaneous emission for leaky and guided modes
    result = st.calc_spe_structure(th_pow=11)
    z = result['z']
    z = st.calc_z_to_lambda(z)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(z, result['leaky']['avg'], label='leaky, air')
    try:
        ax2.plot(z, result['guided']['avg'], label='guided, air')
    except KeyError:
        pass
    spe_air = result['leaky']['avg'] + result['guided']['avg']

    # Medium 2
    # Create structure
    st = LifetimeTmm()
    st.set_vacuum_wavelength(lam0)
    st.add_layer(1.5 * lam0, sio2)
    st.add_layer(d_etds, edts)
    st.add_layer(1.5 * lam0, water)
    st.info()

    # Calculate spontaneous emission for leaky and guided modes
    result = st.calc_spe_structure(th_pow=11)
    z = result['z']
    z = st.calc_z_to_lambda(z)

    # Plot results
    ax1.plot(z, result['leaky']['avg'], label='leaky, water')
    try:
        ax2.plot(z, result['guided']['avg'], label='guided, water')
    except KeyError:
        pass
    spe_water = result['leaky']['avg'] + result['guided']['avg']

    fp = np.mean(spe_water) / np.mean(spe_air)
    print('Purcell Factor: {:e}'.format(fp))

    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        z = st.calc_z_to_lambda(z)
        ax1.axvline(z, color='k', lw=1, ls='--')
        ax2.axvline(z, color='k', lw=1, ls='--')
    # ax1.set_title('Spontaneous emission rate at boundary for semi-infinite media. LHS n=1.57.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_xlabel('Position z ($\lambda$)')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    if SAVE:
        plt.savefig('../Images/T2_purcell_factor')

    fig, ax1 = plt.subplots()
    z = result['z']
    ax1.plot(z, spe_air, label='Air')
    ax1.plot(z, spe_water, label='Water')
    # Plot internal layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        z = st.calc_z_to_lambda(z)
        ax1.axvline(z, color='k', lw=1, ls='--')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax1.set_xlabel('Position z ($\lambda$)')
    # ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.legend(fontsize='small')
    plt.tight_layout()

    if SAVE:
        plt.savefig('../Images/T2_purcell_factor_total')

    plt.show()

if __name__ == "__main__":
    SAVE = True  # Save figs and data? (bool)

    import lifetmm.Methods.journalPlotting

    lifetmm.Methods.journalPlotting.update()

    # Set vacuum wavelength
    lam0 = 1550

    # Film thickness
    d_etds = 980

    # Material refractive index at lam0
    sio2 = 1.442
    edts = 1.56
    air = 1
    water = 1.3183

    # t2()
    # t2_leaky()
    # t2_fig4()
    # t2_guided()
    # t2_spe_vs_n()
    purcell_factor()
