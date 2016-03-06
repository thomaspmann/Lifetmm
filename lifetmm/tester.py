import scipy as sp
import numpy as np

from numpy import pi, exp, sin, inf, sqrt
from scipy.interpolate import interp1d


def I_mat(nj, nk, n0, pol, th_0):
    """
    Calculates the interference matrix between two layers.
    :param nj: First layer refractive index
    :param nk: Second layer refractive index
    :param n0: Refractive index of incident transparent medium
    :param pol: Polarisation of incoming light ('s' or 'p')
    :param th_0: Angle of incidence of light (0 for normal, pi/2 for glancing)
    :return: I-matrix
    """
    # transfer matrix at an interface
    qj = sqrt(nj**2 - n0.real**2 * sin(th_0)**2)
    qk = sqrt(nk**2 - n0.real**2 * sin(th_0)**2)

    if pol == 's':
        r = (qj - qk) / (qj + qk)
        t = (2 * qj) / (qj + qk)

    elif pol == 'p':
        r = (qj * nk**2 - qk * nj**2) / (qj * nk**2 + qk * nj**2)
        t = (2 * nj * nk * qj) / (qj * nk**2 + qk * nj**2)

    else:
        raise ValueError("Polarisation must be 's' or 'p' when angle of incidence is"
                         " not 90$\degree$s")

    if t == 0:
        raise ValueError('Transmission is zero, cannot evaluate I+mat. Check input parameters.')

    return (1/t) * np.array([[1, r], [r, 1]], dtype=complex)


def L_mat(nj, dj, lam_vac, n0, th_0):
    """
    Calculates the propagation.
    :param n: complex dielectric constant
    :param d: thickness
    :param lam_vac: wavelength
    :return:  L-matrix
    """
    qj = sp.sqrt(nj**2 - n0.real**2 * np.sin(th_0)**2)
    eps = (2*pi*qj) / lam_vac
    return np.array([[exp(-1j*eps*dj), 0], [0, exp(1j*eps*dj)]], dtype=complex)


def TransferMatrix(d_list, n_list, lam_vac, th_0, pol, x_step=1):
    """
    Evaluate the transfer matrix over the entire structure.
    :param pol: polarisation of incoming light ('s' or 'p')
    :param n_list: list of refractive indices for each layer (can be complex)
    :param d_list: list of thicknesses for each layer
    :param th_0: angle of incidence (0 for normal, pi/2 for glancing)
    :param lam_vac: vacuum wavelength of light
    :param glass: Bool. If there is a thick superstrate present then make true as interference
                        in this layer is negligible
    :return: Dictionary of all input and output params related to structure
    """
    # convert lists to numpy arrays if they're not already.
    n_list = np.array(n_list, dtype=complex)
    d_list = np.array(d_list, dtype=float)

    # input tests
    if ((hasattr(lam_vac, 'size') and lam_vac.size > 1) or (hasattr(th_0, 'size')
                                                            and th_0.size > 1)):
        raise ValueError('This function is not vectorized; you need to run one '
                         'calculation at a time (1 wavelength, 1 angle, etc.)')
    if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
        raise ValueError("Problem with n_list or d_list!")
    if (d_list[0] != inf) or (d_list[-1] != inf):
        raise ValueError('d_list must start and end with inf!')
    if type(x_step) != int:
        raise ValueError('x_step must be an integer otherwise. Reduce SI unit'
                         'inputs for thicknesses and wavelengths for greater resolution ')
    if th_0 >= pi/2 or th_0 <= -pi/2:
        raise ValueError('The light is not incident on the structure. Check input theta '
                         '(0 <= theta < pi/2')

    # Set first and last layer thicknesses to zero (from inf) - helps for summing later in program
    d_list[0] = 0
    d_list[-1] = 0

    num_layers = np.size(d_list)
    n = n_list
    # Start position of each layer
    d_cumsum = np.cumsum(d_list)
    # x positions to evaluate E field at
    x_pos = np.arange((x_step / 2.0), sum(d_list), x_step)

    # get x_mat - specifies what layer the corresponding point in x_pos is in
    comp1 = np.kron(np.ones((num_layers, 1)), x_pos)
    comp2 = np.transpose(np.kron(np.ones((len(x_pos), 1)), d_cumsum))
    x_mat = sum(comp1 > comp2, 0)

    # calculate the total system transfer matrix S
    # S = I_mat(n[0], n[1], n[0], pol, th_0)
    # for layer in range(1, num_layers - 1):
    #     mI = I_mat(n[layer], n[layer + 1], n[0], pol, th_0)
    #     mL = L_mat(n[layer], d_list[layer], lam_vac, n[0], th_0)
    #     S = S @ mI @ mL

    # calculate the total system transfer matrix S
    S = I_mat(n[0], n[1], n[0], pol, th_0)
    for layer in range(1, num_layers - 1):
        mL = L_mat(n[layer], d_list[layer], lam_vac, n[0], th_0)
        mI = I_mat(n[layer], n[layer + 1], n[0], pol, th_0)
        # S = np.asarray(np.mat(S) * np.mat(mL) * np.mat(mI))
        S = S @ mL
        S = S @ mI

    # JAP Vol 86 p.487 Eq 9 and 10: Power Reflection and Transmission
    R = abs(S[1, 0] / S[0, 0]) ** 2
    T = abs(1 / S[0, 0]) ** 2  # note this is incorrect https://en.wikipedia.org/wiki/Fresnel_equations

    # calculate primed transfer matrices for info on field inside the structure
    E = np.zeros(len(x_pos), dtype=complex)  # Initialise E field
    E_avg = np.zeros(num_layers)
    for layer in range(1, num_layers):
        qj = sp.sqrt(n[layer]**2 - n[0].real**2 * np.sin(th_0)**2)
        eps = (2 * np.pi * qj) / lam_vac
        dj = d_list[layer]
        x_indices = np.where(x_mat == layer)
        # Calculate depth into layer
        x = x_pos[x_indices] - d_cumsum[layer - 1]
        # Calculate S_Prime
        S_prime = I_mat(n[0], n[1], n[0], pol, th_0)
        for layerind in range(2, layer + 1):
            mL = L_mat(n[layerind - 1], d_list[layerind - 1], lam_vac, n[0], th_0)
            mI = I_mat(n[layerind - 1], n[layerind], n[0], pol, th_0)
            # S_prime = np.asarray(np.mat(S_prime) * np.mat(mL) * np.mat(mI))
            S_prime = S_prime @ mL
            S_prime = S_prime @ mI

        # Calculate S_dprime (double prime)
        S_dprime = np.eye(2)
        for layerind in range(layer, num_layers - 1):
            mI = I_mat(n[layerind], n[layerind + 1], n[0], pol, th_0)
            mL = L_mat(n[layerind + 1], d_list[layerind + 1], lam_vac, n[0], th_0)
            # S_dprime = np.asarray(np.mat(S_dprime) * np.mat(mI) * np.mat(mL))
            S_dprime = S_dprime @ mI
            S_dprime = S_dprime @ mL

        # Electric Field Profile
        num = S_dprime[0, 0] * exp(-1j*eps*(dj-x)) + S_dprime[1, 0] * exp(1j*eps*(dj-x))
        den = S_prime[0, 0] * S_dprime[0, 0] * exp(-1j*eps*dj) + S_prime[0, 1] * S_dprime[1, 0] * exp(1j*eps*dj)
        E[x_indices] = num / den

        # Average E field inside the layer
        if not d_list[layer] == 0:
            E_avg[layer] = sum(abs(E[x_indices])**2) / (x_step*d_list[layer])

    # |E|^2
    E_square = abs(E[:]) ** 2

    # Absorption coefficient in 1/cm
    absorption = np.zeros(num_layers)
    for layer in range(1, num_layers):
        absorption[layer] = (4 * np.pi * np.imag(n[layer])) / (lam_vac * 1.0e-7)

    return {'E_square': E_square, 'absorption': absorption, 'x_pos': x_pos,  # output functions of position
            'R': R, 'T': T, 'E': E, 'E_avg': E_avg,  # output overall properties of structure
            'd_list': d_list, 'th_0': th_0, 'n_list': n_list, 'lam_vac': lam_vac, 'pol': pol,  # input structure
            }


class LifetimeTmm:
    def __init__(self, d_list, n_list, x_step=1):
        """
        Initilise with the structure of the material to be simulated
        """
        self.d_list = d_list
        self.n_list = n_list
        self.x_step = x_step

    def setPolarization(self, pol):
        self.pol = pol

    def setAngle(self, th_0, units='radians'):
        if units == 'radians':
            self.th_0 = th_0
        elif units == 'degrees':
            self.th_0 = th_0
        else:
            raise ValueError('Units of angle not recognised.')

    def setWavelength(self, lam):
        self.lam = lam

    def calculate(self):
        d_list = self.d_list
        n_list = self.n_list
        lam_vac = self.lam
        th_0 = self.th_0
        pol = self.pol
        x_step = self.x_step

        result = TransferMatrix(d_list, n_list, lam_vac, th_0, pol, x_step)

        self.x_pos = result['x_pos']
        self.R = result['R']
        self.T = result['T']
        self.E = result['E']
        self.E = result['E']
        self.E_square = result['E_square']
        self.E_avg = result['E_avg']
        self.absorption = result['absorption']


def mcgehee():
    '''
    Copy of the stanford group simulation at 600nm
    '''
    # list of wavelengths to evaluate
    lambda_vac = 600
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm
    d_list = [inf, 110, 35, 220, 7, inf]
    # list of refractive indices
    n_list = [1.4504, 1.7704+0.01161j, 1.4621+0.04426j, 2.12+0.3166016j, 2.095+2.3357j, 1.20252+7.25439j]


    data = np.zeros(sum(d_list[1:-1]))


    runs = 0

    for th in [0]:
        for pol in ['s']:
            runs += 1
            data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol)['E_square'])
    data /= runs

    plt.figure()
    plt.plot(data, label='data')
    dsum = np.cumsum(d_list[1:-1])
    plt.axhline(y=1, linestyle='--', color='k')
    # for i, xmat in enumerate(dsum):
        # plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
        # plt.text(xmat-70, max(data)*0.99,'n: %.2f' % n_list[i+1], rotation=90)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    # plt.title('E-Field Intensity in Device. E_avg in Erbium: %.4f' % E_avg)
    plt.legend()
    plt.show()


def fingdist():
    # list of wavelengths to evaluate
    lambda_vac = [1540]
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)
    # list of layer thicknesses in nm
    d_list = [inf, 1000, 1000, 10, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 1.55, 1, 1.41, 1.41]
    # Initialise plotting data
    dist = []
    data = []
    for i in range(200):  # number of loops
        x = 10 * (i+1)
        dist.append(x)
        d_list[3] = x  # modify the air layer

        E_avg = 0
        runs = 0
        weighting = 0
        for lam in lambda_vac:
            for th in th_0:
                for pol in ['s', 'p']:
                    for rev in [False, True]:
                        runs += 1
                        if rev:
                            weighting = n_list[-1]
                        elif not rev:
                            weighting = n_list[0]

                        weighting *= (np.sin(th * degree))

                        E_avg += (weighting * TransferMatrix(d_list, n_list, lam,
                                                            th * degree, pol, reverse=rev)['E_avg'][1])
        E_avg /= runs
        data.append(E_avg)

    plt.figure()
    plt.plot(dist, data, '.-', label='E')

    plt.xlabel('Air gap between 1um n=1.55 spacer placed on the erbium layer and finger (nm)')
    plt.ylabel('Average Normalized |E|$^2$Intensity over Er layer')
    plt.title('Effect of air gap thickness on the avg E in the erbium layer (n_finger = 1.41)', y=1.08)
    plt.savefig('figs/fingdist5.png')
    plt.show()


def samplePol():
    # list of wavelengths to evaluate
    lambda_vac = [1540]
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)
    # list of layer thicknesses in nm
    d_list = [inf, 1000, 1000, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 3, 3]

    # List of medium refractive indices
    nmed_list = linspace(1, 2, num=500)

    data = []
    for nmed in nmed_list:
        runs = 0
        E_avg = 0
        weighting = 0
        n_list[2] = nmed
        print('Medium n:' + str(nmed))
        for lam in lambda_vac:
            for th in th_0:
                for pol in ['s', 'p']:
                    for rev in [False, True]:
                        runs += 1
                        if rev:
                            weighting = n_list[-1]
                        elif not rev:
                            weighting = n_list[0]

                        weighting *= (1-np.sin(th * degree))

                        E_avg += (weighting * TransferMatrix(d_list, n_list, lam, th * degree, pol, reverse=rev)
                                                                                                        ['E_avg'][1])
        E_avg /= runs
        data.append(E_avg)

    data = np.asarray(data)

    plt.figure()
    plt.plot(nmed_list, data, '.-')
    plt.xlabel('n of medium')
    plt.ylabel('Average |E|$^2$Intensity in Erbium layer')
    plt.title('1000nm erbium, 1000nm sensing medium (n = x axis).')
    plt.savefig('figs/samplePol_1min_sin.png')
    plt.show()


def sample():
    # list of wavelengths to evaluate
    lambda_vac = [1540]
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)
    # list of layer thicknesses in nm
    d_list = [inf, 1000, 100, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 1, 1.41]

    # List of medium refractive indices
    nmed_list = linspace(1.4, 1.42, num=500)

    data = []
    for nmed in nmed_list:
        runs = 0
        E_avg = 0
        weighting = 0
        n_list[-1] = nmed
        print('Medium n:' + str(nmed))
        for lam in lambda_vac:
            for th in th_0:
                for pol in ['s', 'p']:
                    for rev in [False, True]:
                        runs += 1
                        if rev:
                            weighting = n_list[-1]
                        elif not rev:
                            weighting = n_list[0]

                        weighting *= (sin(th * degree))

                        E_avg += (weighting * TransferMatrix(d_list, n_list, lam, th * degree, pol, reverse=rev)
                                                                                                        ['E_avg'][1])
        E_avg /= runs
        data.append(E_avg)

    data = np.asarray(data)

    plt.figure()
    plt.plot(nmed_list, data, '.-')
    plt.xlabel('n of medium')
    plt.ylabel('Average |E|$^2$Intensity in Erbium layer')
    plt.title('1000nm erbium, 100nm air, 1000nm sensing medium (n = x axis).')
    plt.savefig('figs/sample2.png')
    plt.show()


def finger():
    """
    Vary the refractive index of the medium and evaluate the average electric field strength
    inside the erbium layer compared to in bulk
    """
    # Loop parameters
    # list of wavelengths to evaluate
    lambda_vac = [1540]
    # incoming light angle (in degrees)
    th_0 = linspace(0, 90, num=90, endpoint=False)

    # list of layer thicknesses in nm. First and last layer are semi-infinite ambient and substrate layers
    d_list = [inf, 1000, 100, 1E5, inf]
    # list of refractive indices
    n_list = [1.5, 1.5, 1, 1.37, 1.37]

    mM = linspace(0, 20, num=20)
    nRange = [2.73E-5 * x + 1.37 for x in mM]
    # nRange = 1.325 + mM * 2.73E-5

    ydata = np.zeros(len(nRange))
    for i, n in enumerate(nRange):
        print('i is %d and n is %f' % (i, n))
        E_avg = 0
        runs = 0
        n_list[4] = n
        print('i is %d and n is %f' % (i, n_list[4]))
        for th in th_0:
            for pol in ['s', 'p']:
                for rev in [True, False]:
                    runs += 1
                    # data += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_square'] /
                    #          TransferMatrix(d_list, n_listB, lambda_vac, th * degree, pol, reverse=rev)['E_square'])

                    E_avg += (TransferMatrix(d_list, n_list, lambda_vac, th * degree, pol, reverse=rev)['E_avg'][1])
        ydata[i] = E_avg/runs

    plt.figure()
    plt.plot(mM, 1/ydata)
    plt.xlabel('Glucose concentration in water (mM)')
    plt.ylabel('Lifetime (1/E^2)')
    plt.title('Average E-Field Intensity in Device')
    plt.savefig('figs/finger.png')
    plt.show()