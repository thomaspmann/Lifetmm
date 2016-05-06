import matplotlib.pyplot as plt
from lifetmm import *
from numpy import pi, linspace, inf, array, sum, cos, sin
from scipy.interpolate import interp1d

# # To run a sample use the following in python console:
# import lifetmm.examples; lifetmm.examples.sample1()

# "5 * degree" is 5 degrees expressed in radians
# "1.2 / degree" is 1.2 radians expressed in degrees
degree = pi / 180
mmTOnm = 1E6


def mcgehee():
    st = LifetimeTmm()
    st.add_layer(0, 1.4504)
    st.add_layer(110, 1.7704+0.01161j)
    st.add_layer(35, 1.4621+0.04426j)
    st.add_layer(220, 2.12+0.3166016j)
    st.add_layer(7, 2.095+2.3357j)
    st.add_layer(200, 1.20252 + 7.25439j)
    st.add_layer(0, 1.20252 + 7.25439j)

    st.set_wavelength(600)
    st.set_polarization('s')
    st.set_angle(0, units='degrees')

    y = st.structure_E_field()['E_square']

    plt.figure()
    plt.plot(y)
    plt.axhline(y=1, linestyle='--', color='k')
    for z in st.get_layer_boundaries():
        plt.axvline(x=z, color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.show()


def spe():
    # Create structure
    st = LifetimeTmm()
    # st.add_layer(1550, 1)
    st.add_layer(1550, 3.48)
    # st.add_layer(1000, 1)
    # st.add_layer(1000, 8)
    st.add_layer(1550, 1)
    # st.add_layer(1550, 3.48)

    # Set light info
    st.set_wavelength(1550)

    # Get results
    result = st.spe_structure()
    z = result['z']
    spe_TE = result['spe_TE']
    spe_TM_h = result['spe_TM_h']
    spe_TM_v = result['spe_TM_v']

    # Plot s
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(z, spe_TE, label='TE')
    ax2 = fig.add_subplot(312)
    ax2.plot(z, spe_TM_h, label='TM')
    ax3 = fig.add_subplot(313)
    ax3.plot(z, spe_TM_v, label='TM')

    ax1.set_title('Spontaneous Emission Rate')
    ax1.set_ylabel('Purcell Factor')
    ax3.set_xlabel('Position in layer (nm)')

    ax1.axhline(y=1, linestyle='--', color='k')
    # ax2.axhline(y=1, linestyle='--', color='k')
    # Plot layer boundaries
    for z in st.get_layer_boundaries():
        ax1.axvline(z, color='r', lw=2)
        ax2.axvline(z, color='r', lw=2)
        ax3.axvline(z, color='r', lw=2)
    ax1.legend(title='Horizontal Dipoles')
    ax2.legend(title='Horizontal Dipoles')
    ax3.legend(title='Vertical Dipoles')
    plt.ylim([0,100])
    plt.show()


def test_symmetry():
    # Create structure
    st = LifetimeTmm()
    st.add_layer(1000, 3.48)
    # st.add_layer(2000, 3.48)
    st.add_layer(1000, 1)
    # st.add_layer(100, 1)
    # st.add_layer(1000, 3.48)

    # Set light info
    st.set_wavelength(1550)
    st.set_polarization('s')
    st.set_angle(70, units='degrees')

    print('Lower')
    y_lower = st.structure_E_field(radiative='Lower', time_rev=True)['E_square']

    print('Upper')
    theta = st.snell(st.n_list[0], st.n_list[-1], st.th)
    # theta = np.conj(theta)
    st.th = theta
    st.flip()
    y = st.structure_E_field(radiative='Upper', time_rev=True)['E_square']
    y_upper = y[::-1]
    st.flip()

    plt.figure()
    plt.plot(y_lower, label='Lower')
    plt.plot(y_upper, label='Upper', ls='--', color='r')
    plt.axhline(y=1, linestyle='--', color='k')
    for z in st.get_layer_boundaries():
        plt.axvline(x=z, color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.legend()
    plt.show()


def lower_vs_upper():
    # Create structure
    st = LifetimeTmm()
    # st.add_layer(1550, 1)
    st.add_layer(1550, 3.48)
    st.add_layer(1550, 1)
    # st.add_layer(1550, 3.48)

    # Set light info
    st.set_wavelength(1550)
    st.set_polarization('s')
    theta = 50
    st.set_angle(theta, units='degrees')

    print('Lower')
    y_lower = st.structure_E_field(radiative='Lower', time_rev=False)['E_square']
    # y_lower = st.structure_E_field(radiative='Lower', time_rev=True)['E_square']
    print('Upper')
    # theta = st.snell(st.n_list[0], st.n_list[-1], st.th)
    # theta = np.conj(theta)
    # print(theta)
    # st.th = theta
    # y_upper = 0
    y_upper = st.structure_E_field(radiative='Upper', time_rev=False)['E_square']
    # y_upper = st.structure_E_field(radiative='Upper', time_rev=True)['E_square']

    plt.figure()
    plt.plot(y_lower, label='Lower')
    plt.plot(y_upper, label='Upper', color='g')
    plt.axhline(y=1, linestyle='--', color='k')
    for z in st.get_layer_boundaries():
        plt.axvline(x=z, color='r', lw=2)
    plt.axvline(x=z, color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.title('Angle of incidence {} degrees'.format(theta))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # mcgehee()
    # test_symmetry()
    # lower_vs_upper()
    spe()
