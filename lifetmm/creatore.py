import scipy as sp
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from numpy import pi, exp, sin, cos, sqrt
from tqdm import *


class LifetimeTmm:
    def __init__(self):
        self.d_list = np.array([], dtype=float)
        self.n_list = np.array([], dtype=complex)
        self.d_cumsum = np.array([], dtype=float)
        self.x_step = 1
        self.lam_vac = 0
        self.num_layers = 0
        self.pol = 'u'
        self.th = 0

    def add_layer(self, d, n):
        self.d_list = np.append(self.d_list, d)
        self.n_list = np.append(self.n_list, n)
        self.d_cumsum = np.cumsum(self.d_list)
        self.num_layers = np.size(self.d_list)

    def set_wavelength(self, lam_vac):
        self.lam_vac = lam_vac

    def set_polarization(self, pol):
        if pol not in ['s', 'p']:
            raise ValueError("Polarisation must be 's' or 'p' when angle of incidence is"
                             " not 90$\degree$s")
        self.pol = pol

    def set_angle(self, th, units='radians'):
        if hasattr(th, 'size') and th.size > 1:
            raise ValueError('This function is not vectorized; you need to run one '
                             'calculation for each angle at a time')
        if units == 'radians':
            assert 0 <= th < pi/2, 'The light is not incident on the structure. ' \
                                     'Check input theta satisfies -pi/2 <= theta < pi/2'
            self.th = th
        elif units == 'degrees':
            assert 0 <= th < 90, 'The light is not incident on the structure. ' \
                              'Check input theta satisfies -90 <= theta < 90'
            self.th = th * (pi/180)
        else:
            raise ValueError('Units of angle not recognised. Please enter \'radians\' or \'degrees\'.')

    @staticmethod
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
        return sp.arcsin(np.real_if_close(n_1*sin(th_1) / n_2))

    @staticmethod
    def q(nj, n0, th):
        return sqrt(nj**2 - (n0.real*sin(th))**2)

    def I_mat(self, nj, nk):
        n0 = self.n_list[0]
        qj = self.q(nj, n0, self.th)
        qk = self.q(nk, n0, self.th)
        if self.pol is 'p':
            r = (qj * nk**2 - qk * nj**2) / (qj * nk**2 + qk * nj**2)
            t = (2 * nj * nk * qj) / (qj * nk**2 + qk * nj**2)
        else:  # pol is 's' (or 'u')
            r = (qj - qk) / (qj + qk)
            t = (2 * qj) / (qj + qk)
        assert t != 0, ValueError('Transmission is zero, cannot evaluate I_mat.')
        return (1/t) * np.array([[1, r], [r, 1]], dtype=complex)

    def L_mat(self, nj, dj):
        n0 = self.n_list[0]
        qj = self.q(nj, n0, self.th)
        eps = (2*pi*qj) / self.lam_vac
        delta = eps*dj
        return np.array([[exp(-1j*delta), 0], [0, exp(1j*delta)]], dtype=complex)

    def _simulation_test(self):
        x_step = self.x_step
        if (self.d_list[0] != 0) or (self.d_list[-1] != 0):
            raise ValueError('Structure must start and end with 0!')
        if type(x_step) != int:
            raise ValueError('x_step must be an integer. Reduce SI unit'
                             'inputs for thicknesses and wavelengths for greater resolution ')

    def flip(self):
        """ Flip the structure front-to-back.
        """
        self.d_list = self.d_list[::-1]
        self.n_list = self.n_list[::-1]
        self.d_cumsum = np.cumsum(self.d_list)

    def calc_s_matrix(self):
        d_list = self.d_list
        n = self.n_list
        # calculate the total system transfer matrix S
        S = self.I_mat(n[0], n[1])
        for layer in range(1, self.num_layers - 1):
            mL = self.L_mat(n[layer], d_list[layer])
            mI = self.I_mat(n[layer], n[layer + 1])
            S = S @ mL @ mI
        return S

    def calc_s_primed(self, layer):
        d_list = self.d_list
        n = self.n_list
        # Calculate S_Prime
        S_prime = self.I_mat(n[0], n[1])
        for v in range(1, layer):
            mL = self.L_mat(n[v], d_list[v])
            mI = self.I_mat(n[v], n[v + 1])
            S_prime = S_prime @ mL @ mI
        # Calculate S_dPrime (doubled prime)
        S_dprime = self.I_mat(n[layer], n[layer + 1])
        for v in range(layer + 1, self.num_layers - 1):
            mL = self.L_mat(n[v], d_list[v])
            mI = self.I_mat(n[v], n[v + 1])
            S_dprime = S_dprime @ mL @ mI
        return S_prime, S_dprime

    @staticmethod
    def matrix_2x2_determinant(matrix):
        return matrix[0, 1]*matrix[1, 0] - matrix[0, 0]*matrix[1, 1]

    def layer_E_field(self, layer, time_reversal=False, pr=False):
        self._simulation_test()

        d_list = self.d_list
        n = self.n_list
        x_step = self.x_step
        lam_vac = self.lam_vac
        num_layers = self.num_layers

        # calculate the transfer matrices
        S = self.calc_s_matrix()
        S_prime, S_dprime = self.calc_s_primed(layer)

        # Wavevector components in layer
        kj = (2*pi*n[layer]) / lam_vac
        qj = self.q(n[layer], n[0], self.th)
        kj_z = (2 *pi*qj) / lam_vac
        kj_parallel = kj * sin(self.th)

        x = np.arange((x_step / 2.0), d_list[layer], x_step)

        #  Electric Field Profile
        det_S_prime = self.matrix_2x2_determinant(S_prime)
        if not time_reversal:
            rR = S[1, 0] / S[0, 0]
            # In units of W_0
            Wj = (S_prime[1, 1] - rR * S_prime[0, 1]) / det_S_prime
            Xj = (rR * S_prime[0, 0] - S_prime[1, 0]) / det_S_prime
            E = Wj*exp(1j*kj_z*x) + Xj*exp(-1j*kj_z*x)
        else:  # Time reversal
            # Time reversal
            kj_z = np.conj(kj_z)
            rR = S[0, 1] / S[1, 1]
            # # In units of X_0
            # X_0 = 1 / n[0].real
            Wj = (S_prime[0, 1] - rR*S_prime[1, 1]) / det_S_prime
            Xj = (rR*S_prime[1, 0] - S_prime[0, 0]) / det_S_prime
            E = Wj*exp(1j*kj_z*x) + Xj*exp(-1j*kj_z*x)

        E_square = abs(E[:])**2
        E_avg = sum(E_square) / (x_step*d_list[layer])

        return {'x': x, 'E': E, 'E_square': E_square, 'E_avg': E_avg}

    # def dipole_rate(self, E):
    #     if self.pol == 'p':
    #         # Horizontal dipoles (parallel to interface)
    #         E_horizontal = E * kj_z
    #         # Vertical dipoles (perpendicular to interface)
    #         E_vertical = E * kj_parallel
    #     else:  # self.pol == 's':
    #         E_horizontal = E
    #         E_vertical = 0

    def spe_layer(self, layer):
        n = self.n_list

        resolution = 2 ** 11 + 1
        th_emission, dth = np.linspace(0, pi / 2, resolution, endpoint=False, retstep=True)
        x = np.arange((self.x_step / 2.0), self.d_list[layer], self.x_step)
        E_square_theta_lower = np.zeros((resolution, len(x)), dtype=float)
        E_square_theta_upper = np.zeros((resolution, len(x)), dtype=float)
        # Params for tqdm progress bar
        kwargs = {
            'total': resolution,
            'unit': 'theta modes',
            'unit_scale': True,
        }
        for i, th in tqdm(enumerate(th_emission), **kwargs):
            self.set_angle(th)

            # Wavevector components in layer
            k_j = (2*pi*n[layer]) / self.lam_vac

            assert n[0] >= n[-1], 'Refractive index of lower cladding must be larger than the upper cladding'

            # Evaluate for outgoing wave in lower cladding
            E = self.layer_E_field(layer=layer, time_reversal=True)['E']
            # Normalise W.R.T. X_0
            X_0 = 1 / n[0].real
            E *= X_0
            E_square = abs(E[:]) ** 2
            E_square_theta_lower[i, :] = E_square * sin(th)

            # # Evaluate for outgoing wave in upper cladding
            # self.flip()
            # E = self.layer_E_field(layer=self.num_layers-1-layer, time_reversal=True)['E']
            # # Normalise W.R.T. X_0
            # X_0 = 1 / self.n_list[0].real
            # E *= X_0
            # self.flip()
            # E = E[::-1]
            # E_square = abs(E[:]) ** 2
            # E_square_theta_upper[i, :] += E_square * sin(th)

        # Evaluate integral
        integral = integrate.romb(E_square_theta_lower, dx=dth, axis=0)
        # integral += integrate.romb(E_square_theta_upper, dx=dth.real, axis=0)
        spe = integral * n[layer].real**3
        return {'x': x, 'spe': spe}

    def spe_structure(self):
        # x positions to evaluate E field at over entire structure
        x_pos = np.arange((self.x_step / 2.0), sum(self.d_list), self.x_step)
        # get x_mat - specifies what layer the corresponding point in x_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), x_pos)
        comp2 = np.transpose(np.kron(np.ones((len(x_pos), 1)), self.d_cumsum))
        x_mat = sum(comp1 > comp2, 0)
        # Evaluate spontaneous emission rate for each medium inside cladding layers
        spe = np.zeros(len(x_pos), dtype=float)
        for layer in range(1, self.num_layers-1):
            # Calculate x indices inside structure for the layer
            x_indices = np.where(x_mat == layer)
            spe_layer = self.spe_layer(layer=layer)['spe']
            spe[x_indices] = spe_layer
        return {'x': x_pos, 'spe': spe}

    def structure_E_field(self, time_reversal=False):
        # x positions to evaluate E field at over entire structure
        x_pos = np.arange((self.x_step / 2.0), sum(self.d_list), self.x_step)
        # get x_mat - specifies what layer the corresponding point in x_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), x_pos)
        comp2 = np.transpose(np.kron(np.ones((len(x_pos), 1)), self.d_cumsum))
        x_mat = sum(comp1 > comp2, 0)
        # Evaluate spontaneous emission rate for each medium inside cladding layers
        E_square = np.zeros(len(x_pos), dtype=float)
        for layer in range(1, self.num_layers-1):
            # Calculate x indices inside structure for the layer
            x_indices = np.where(x_mat == layer)
            E_layer = self.layer_E_field(layer=layer, time_reversal=time_reversal)['E_square']
            E_square[x_indices] = E_layer
        return {'x': x_pos, 'E_square': E_square}


def mcgehee():
    st = LifetimeTmm()
    st.add_layer(0, 1.4504)
    st.add_layer(110, 1.7704+0.01161j)
    st.add_layer(35, 1.4621+0.04426j)
    st.add_layer(220, 2.12+0.3166016j)
    st.add_layer(7, 2.095+2.3357j)
    # st.add_layer(200, 1.20252 + 7.25439j)
    st.add_layer(0, 1.20252 + 7.25439j)

    st.set_wavelength(600)
    st.set_polarization('s')
    st.set_angle(0)

    y = st.structure_E_field(time_reversal=True)['E_square']

    plt.figure()
    plt.plot(y)
    dsum = getattr(st, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, xmat in enumerate(dsum):
        plt.axvline(x=xmat, linestyle='-', color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.show()


def spe():
    st = LifetimeTmm()

    st.add_layer(0, 3.48)
    st.add_layer(2000, 3.48)
    st.add_layer(2000, 1)
    st.add_layer(0, 1)

    st.set_wavelength(1550)
    st.set_polarization('s')

    result = st.spe_structure()
    # result = st.spe_rate_structure()
    y = result['spe']

    x = result['x']

    plt.figure()
    plt.plot(x, y)
    plt.axhline(y=1, linestyle='--', color='k')
    plt.xlabel('Position in layer (nm)')
    plt.ylabel('Purcell Factor')
    plt.show()

if __name__ == "__main__":
    # spe()
    mcgehee()
