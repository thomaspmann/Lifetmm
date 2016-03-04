import scipy as sp
import numpy as np

from numpy import pi, exp, sin, inf, sqrt
# from scipy.interpolate import interp1d


def thetaCritical(m, n_list):
    """
    :param m: layer containing the emitting atom
    :param n_list: list of refractive indices of the layers
    :return: Return the angle at which TIRF occurs between the layer containing the atom and the cladding with
    the largest refractive index, or pi/2, whichever comes first.
    """

    # Evaluate largest refractive index of claddings
    n_clad = max(n_list[0], n_list[-1])

    # Using Snell's law evaluate the critical angle or return pi/2 if does not exist
    if n_clad/n_list[m] < 1:
        return np.arcsin(n_clad/n_list[m])
    else:
        return pi/2


def snell(n_1, n_2, th_1):
    """
    return angle theta in layer 2 with refractive index n_2, assuming
    it has angle th_1 in layer with refractive index n_1. Use Snell's law. Note
    that "angles" may be complex!!
    """
    # Important that the arcsin here is scipy.arcsin, not numpy.arcsin!! (They
    # give different results e.g. for arcsin(2).)
    # Use real_if_close because e.g. arcsin(2 + 1e-17j) is very different from
    # arcsin(2) due to branch cut
    return sp.arcsin(np.real_if_close(n_1*np.sin(th_1) / n_2))


def H(theta):
    if np.isreal(theta):
        return 1
    else:
        return 0



class LifetimeTmm:
    def __init__(self, d_list, n_list, x_step=1):
        """
        Initilise with the structure of the material to be simulated
        """
        # convert lists to numpy arrays if they're not already.
        n_list = np.array(n_list, dtype=complex)
        d_list = np.array(d_list, dtype=float)
        # input tests
        if (d_list[0] != inf) or (d_list[-1] != inf):
            raise ValueError('d_list must start and end with inf!')
        if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
            raise ValueError("Problem with n_list or d_list!")
        if type(x_step) != int:
            raise ValueError('x_step must be an integer otherwise. Reduce SI unit'
                             'inputs for thicknesses and wavelengths for greater resolution ')
        self.d_list = d_list
        self.n_list = n_list
        self.x_step = x_step

    def setPolarization(self, pol):
        self.pol = pol

    def setAngle(self, th_0, units='radians'):

        if hasattr(th_0, 'size') and th_0.size > 1:
            raise ValueError('This function is not vectorized; you need to run one '
                             'calculation at a time (1 wavelength, 1 angle, etc.)')

        if th_0 >= pi/2 or th_0 <= -pi/2:
            raise ValueError('The light is not incident on the structure. Check input theta '
                             '(0 <= theta < pi/2')

        if units == 'radians':
            self.th_0 = th_0
        elif units == 'degrees':
            self.th_0 = th_0
        else:
            raise ValueError('Units of angle not recognised.')

    def setWavelength(self, lam_vac):
        # input tests
        if hasattr(lam_vac, 'size') and lam_vac.size > 1:
            raise ValueError('This function is not vectorized; you need to run one '
                             'calculation at a time (1 wavelength, 1 angle, etc.)')
        self.lam_vac = lam_vac

    def calculate(self):
        result = LifetimeTmm.TransferMatrix(self)

        self.x_pos = result['x_pos']
        self.R = result['R']
        self.T = result['T']
        self.E = result['E']
        self.E = result['E']
        self.E_square = result['E_square']
        self.E_avg = result['E_avg']
        self.absorption = result['absorption']

    def I_mat(self, nj, nk):

        pol = self.pol
        th_0 = self.th_0
        n0 = self.n_list[0]

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

    def L_mat(self, nj, dj):
        lam_vac = self.lam_vac
        n0 = self.n_list[0]
        th_0 = self.th_0
        qj = sp.sqrt(nj**2 - n0.real**2 * np.sin(th_0)**2)
        eps = (2*pi*qj) / lam_vac
        return np.array([[exp(-1j*eps*dj), 0], [0, exp(1j*eps*dj)]], dtype=complex)

    def TransferMatrix(self):
        d_list = self.d_list
        n_list = self.n_list
        lam_vac = self.lam_vac
        th_0 = self.th_0
        pol = self.pol
        x_step = self.x_step

        # Set first and last layer thicknesses to zero (from inf) - helps for summing later in program
        d_list[0] = 0
        d_list[-1] = 0

        num_layers = np.size(d_list)
        n = n_list
        # Start position of each layer
        d_cumsum = np.cumsum(d_list)
        self.d_cumsum = d_cumsum
        # x positions to evaluate E field at
        x_pos = np.arange((x_step / 2.0), sum(d_list), x_step)

        # get x_mat - specifies what layer the corresponding point in x_pos is in
        comp1 = np.kron(np.ones((num_layers, 1)), x_pos)
        comp2 = np.transpose(np.kron(np.ones((len(x_pos), 1)), d_cumsum))
        x_mat = sum(comp1 > comp2, 0)

        # calculate the total system transfer matrix S
        S = self.I_mat(n[0], n[1])
        for layer in range(1, num_layers - 1):
            mL = self.L_mat(n[layer], d_list[layer])
            mI = self.I_mat(n[layer], n[layer + 1])
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
            S_prime = self.I_mat(n[0], n[1])
            for layerind in range(2, layer + 1):
                mL = self.L_mat(n[layerind - 1], d_list[layerind - 1])
                mI = self.I_mat(n[layerind - 1], n[layerind])
                # S_prime = np.asarray(np.mat(S_prime) * np.mat(mL) * np.mat(mI))
                S_prime = S_prime @ mL
                S_prime = S_prime @ mI

            # Calculate S_dprime (double prime)
            S_dprime = np.eye(2)
            for layerind in range(layer, num_layers - 1):
                mI = self.I_mat(n[layerind], n[layerind + 1])
                mL = self.L_mat(n[layerind + 1], d_list[layerind + 1])
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
