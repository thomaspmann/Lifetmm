import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from lifetmm.Methods.TransferMatrix import TransferMatrix
from lifetmm.Methods.SpontaneousEmissionRate import LifetimeTmm
from tqdm import tqdm


def mcgehee():
    st = TransferMatrix()
    st.add_layer(0, 1.4504)
    st.add_layer(110, 1.7704+0.01161j)
    st.add_layer(35, 1.4621+0.04426j)
    st.add_layer(220, 2.12+0.3166016j)
    st.add_layer(7, 2.095+2.3357j)
    st.add_layer(200, 1.20252 + 7.25439j)
    st.add_layer(0, 1.20252 + 7.25439j)

    plt.figure()
    st.set_wavelength(600)
    st.set_polarization('s')
    st.set_angle(0, units='degrees')

    y = st.structure_field()['A_squared']
    plt.plot(y)

    plt.axhline(y=1, linestyle='--', color='k')
    for z in st.get_layer_boundaries()[:-1]:
        plt.axvline(x=z, color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.savefig('../Images/McGehee_structure.png', dpi=300)
    plt.show()


def spe():
    # Create structure
    st = LifetimeTmm()

    # Set vacuum wavelength
    lam0 = 1550
    st.set_wavelength(lam0)

    # Add layers
    # st.add_layer(lam0, 1)
    st.add_layer(lam0, 3.48)
    st.add_layer(lam0, 1)
    st.add_layer(lam0, 3.48)
    # st.add_layer(lam0, 1)

    # Get results
    result = st.spe_structure()
    z = result['z']
    spe = result['spe']
    spe_TE = spe['TE_total']
    spe_TM_p = spe['TM_p_total']
    spe_TM_s = spe['TM_s_total']

    # Plot spe rates
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(z, spe_TE, label='TE')
    ax1.plot(z, spe_TM_p, label='TM')
    ax1.plot(z, spe_TE+spe_TM_p, 'k', label='TE + TM')
    ax2 = fig.add_subplot(212)
    ax2.plot(z, spe_TM_s, label='TM')

    ax1.set_title('Spontaneous Emission Rate. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('Position in layer (nm)')

    ax1.axhline(y=1, linestyle='--', color='k')
    ax2.axhline(y=1, linestyle='--', color='k')
    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(z, color='r', lw=2)
        ax2.axvline(z, color='r', lw=2)
    ax1.legend(title='Horizontal Dipoles')
    ax2.legend(title='Vertical Dipoles')
    plt.show()


def test1():
    # Create structure
    st = TransferMatrix()

    # Set vacuum wavelength
    lam0 = 1550
    st.set_wavelength(lam0)
    st.set_polarization('TE')
    st.set_field('E')
    deg = 60
    st.set_angle(np.deg2rad(deg))
    # Add layers
    st.add_layer(20 * lam0, 1)
    st.add_layer(4 * lam0, 3.48)
    st.add_layer(2 * lam0, 1)

    # Get results
    result = st.structure_field()
    z = result['z']
    field = result['A_squared']

    # Plot spe rates
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(z/lam0, field, label=('TE, ' + str(deg) + 'deg'))

    ax1.set_title('Semi-infinite')
    ax1.set_xlabel('$z/\lambda_0$')
    ax1.set_ylabel('$|E|^2$')
    ax1.axhline(y=1, linestyle='--', color='k')
    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(z/lam0, color='r', lw=2)
    ax1.legend(title='Horizontal Dipoles')
    plt.show()


def guiding():
    # Create structure
    st = LifetimeTmm()

    # Set vacuum wavelength
    lam0 = 1550
    st.set_wavelength(lam0)
    st.set_polarization('s')
    # Add layers
    st.add_layer(2.5 * lam0, 1)
    st.add_layer(1*lam0, 3.48)
    st.add_layer(2.5 * lam0, 1)

    # Get results
    result = st.spe_structure()
    z = result['z']
    spe = result['spe']
    spe_TE = spe['TE_total']
    spe_TM_p = spe['TM_p_total']
    spe_TM_s = spe['TM_s_total']

    # Plot spe rates for radiative modes
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex='col', sharey='col', figsize=(15, 7))
    # ax1 = fig.add_subplot(211)
    ax1.plot(z, spe_TE, label='TE')
    ax1.plot(z, spe_TM_p, label='TM')
    ax1.plot(z, spe_TE+spe_TM_p, 'k', label='TE + TM')
    ax2.plot(z, spe_TM_s, label='TM')

    ax1.set_title('Radiative Modes. LHS n=3.48, RHS n=1.')
    ax1.set_ylabel('$\Gamma / \Gamma_0$')
    ax2.set_ylabel('$\Gamma /\Gamma_0$')
    ax2.set_xlabel('Position in layer (nm)')

    ax1.axhline(y=1, linestyle='--', color='k')
    ax2.axhline(y=1, linestyle='--', color='k')
    # Plot layer boundaries
    for z in st.get_layer_boundaries()[:-1]:
        ax1.axvline(z, color='r', lw=2)
        ax2.axvline(z, color='r', lw=2)
    ax1.legend(title='Horizontal Dipoles')
    ax2.legend(title='Vertical Dipoles')

    # Find Guided modes
    t_list = []
    t_list_r = []
    t_list_i = []
    th_list = []
    for th in np.linspace(0, pi/2, num=500, endpoint=False):
        st.set_angle(th)
        S_mat = st.S_mat()
        t = S_mat[0, 0]
        # t = np.real_if_close([t])
        # t = t[0]
        # if np.isclose(abs(t), [1]):
        #     print('t is {0:g}, magnitude is {1:g} at {2:g}'.format(t, abs(t), np.rad2deg(th)))
        th_list.append(th*(180/pi))
        t = 1/t
        t_list.append(abs(t))
        t_list_r.append(t.real)
        t_list_i.append(t.imag)
    ax4.plot(th_list, t_list, label='Magnitude')
    ax4.plot(th_list, t_list_r, label='Real')
    ax4.plot(th_list, t_list_i, label='Imaginary')
    ax4.set_xlabel('Theta (Degrees)')
    ax4.set_ylabel('Transmission ($1/s_22 = abs(t))$')
    ax4.legend()
    plt.show()


def guiding2():
    # Create structure
    st = LifetimeTmm()
    # Set vacuum wavelength
    lam0 = 1550
    st.set_wavelength(lam0)
    # Add layers
    st.set_polarization('p')
    st.set_field('E')
    st.add_layer(0, 3.48)
    # st.add_layer(2.5 * lam0, 1)
    st.add_layer(1 * lam0, 1)
    # st.add_layer(2.5 * lam0, 1)
    st.add_layer(0, 3.48)

    # Find Guided modes
    t_list = np.array([])
    th_list = np.array([])
    for th in tqdm(np.linspace(0, 90, num=500, endpoint=False)):
        st.set_angle(th, units='degrees')
        k, q, k_11 = st.wave_vector(2)
        print(q)

        S_mat = st.S_mat()
        t = S_mat[0, 0]
        # if np.isclose(t, [0]):
        #     print('close {}'.format(t))
        th_list = np.append(th_list, th)
        t_list = np.append(t_list, t)

    # Plot S_11 rates for radiative modes
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(th_list, abs(t_list), label='Magnitude')
    ax1.plot(th_list, t_list.real, label='Real')
    ax1.plot(th_list, t_list.imag, label='Imaginary')
    ax1.set_ylabel('Transmission ($s_{22}$)')
    ax1.legend()
    t_list = 1/t_list
    ax2.plot(th_list, abs(t_list), label='Magnitude')
    ax2.plot(th_list, t_list.real, label='Real')
    ax2.plot(th_list, t_list.imag, label='Imaginary')
    ax2.set_xlabel('Theta (Degrees)')
    ax2.set_ylabel('Transmission ($t = 1/s_{22}$)')
    ax2.legend()
    ax1.axhline(color='k')
    ax2.axhline(color='k')
    plt.show()


def guiding3():
    # Create structure
    st = LifetimeTmm()
    lam0 = 1550
    st.set_wavelength(lam0)
    st.set_polarization('TE')
    st.set_field('E')
    air = 1
    sio2 = 3.48
    st.add_layer(0 * lam0, air)
    st.add_layer(1 * lam0, sio2)
    st.add_layer(0 * lam0, air)

    # Find Guided modes
    t_list = np.array([])
    th_list = np.array([])

    st.set_angle(80, units='degrees')
    k, q, k_11 = st.wave_vector(2)
    print(q)

    S_mat = st.S_mat()
    t = S_mat[0, 0]
    th_list = np.append(th_list, th)
    t_list = np.append(t_list, t)

    # Plot S_11 rates for radiative modes
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='none')
    ax1.plot(th_list, abs(t_list), label='Magnitude')
    ax1.plot(th_list, t_list.real, label='Real')
    ax1.plot(th_list, t_list.imag, label='Imaginary')
    ax1.set_ylabel('Transmission ($s_{22}$)')
    ax1.legend()
    t_list = 1/t_list
    ax2.plot(th_list, abs(t_list), label='Magnitude')
    ax2.plot(th_list, t_list.real, label='Real')
    ax2.plot(th_list, t_list.imag, label='Imaginary')
    ax2.set_xlabel('Theta (Degrees)')
    ax2.set_ylabel('Transmission ($t = 1/s_{22}$)')
    ax2.legend()
    ax1.axhline(color='k')
    ax2.axhline(color='k')
    plt.show()

if __name__ == "__main__":
    # mcgehee()
    # spe()
    guiding2()

