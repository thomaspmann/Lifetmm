import scipy as sp
import numpy as np
import scipy.integrate as integrate

from numpy import pi, exp, sin, sqrt, sum
from tqdm import *
import matplotlib.pyplot as plt


class LifetimeTmm:
    def __init__(self):
        self.d_list = np.array([], dtype=float)
        self.n_list = np.array([], dtype=complex)
        self.d_cumsum = np.array([], dtype=float)
        self.z_step = 1
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
        return np.array([[exp(-1j*eps*dj), 0], [0, exp(1j*eps*dj)]], dtype=complex)

    def _simulation_test(self):
        if (self.d_list[0] != 0) or (self.d_list[-1] != 0):
            raise ValueError('Structure must start and end with 0!')
        if type(self.z_step) != int:
            raise ValueError('z_step must be an integer. Reduce SI unit'
                             'inputs for thicknesses and wavelengths for greater resolution ')

    def flip(self):
        """ Flip the structure front-to-back.
        """
        self.d_list = self.d_list[::-1]
        self.n_list = self.n_list[::-1]
        self.d_cumsum = np.cumsum(self.d_list)

    def s_mat(self):
        d_list = self.d_list
        n = self.n_list
        # calculate the total system transfer matrix S
        S = self.I_mat(n[0], n[1])
        for layer in range(1, self.num_layers - 1):
            mL = self.L_mat(n[layer], d_list[layer])
            mI = self.I_mat(n[layer], n[layer + 1])
            S = S @ mL @ mI
        return S

    def s_primed_mat(self, layer):
        d_list = self.d_list
        n = self.n_list
        # Calculate S_Prime
        S_prime = self.I_mat(n[0], n[1])
        for v in range(1, layer):
            mL = self.L_mat(n[v], d_list[v])
            mI = self.I_mat(n[v], n[v + 1])
            S_prime = S_prime @ mL @ mI
        return S_prime

    def s_dprimed_mat(self, layer):
        d_list = self.d_list
        n = self.n_list
        # Calculate S_dPrime (doubled prime)
        S_dprime = self.I_mat(n[layer], n[layer + 1])
        for v in range(layer + 1, self.num_layers - 1):
            mL = self.L_mat(n[v], d_list[v])
            mI = self.I_mat(n[v], n[v + 1])
            S_dprime = S_dprime @ mL @ mI
        return S_dprime

    @staticmethod
    def matrix_2x2_determinant(matrix):
        return matrix[0, 1]*matrix[1, 0] - matrix[0, 0]*matrix[1, 1]

    def calculate_coefficients(self, radiative='lower'):
        n = self.n_list
        # calculate the transfer matrices
        S = self.s_mat()

        X = np.zeros(self.num_layers, dtype=complex)
        W = np.zeros(self.num_layers, dtype=complex)

        if radiative == 'lower':
            # Layer coefficients (time reversal BCs)
            X[0] = 1 / n[0]
            rR = S[0, 1] / S[1, 1]
            W[0] = X[0] * rR

            for j in range(1, self.num_layers - 1):
                X[j] = 1

        return W, X

    def lower_cladding_E_field(self, W0, X0):
        n = self.n_list

        # Wave vector components in layer
        qj = self.q(n[0], n[0], self.th)
        kj_z = (2 * pi * qj) / self.lam_vac

        # z positions to evaluate E at
        z = np.arange((-self.d_list[0]+self.z_step / 2.0), 0, self.z_step)

        # Evaluate E field
        E = W0 * exp(-1j * kj_z * z) + X0 * exp(1j * kj_z * z)
        return {'z': z, 'E': E}

    def upper_cladding_E_field(self, Wmp1, Xmp1):
        n = self.n_list

        # Wave vector components in layer
        qj = self.q(n[-1], n[0], self.th)
        kj_z = (2 * pi * qj) / self.lam_vac

        # Time reversed
        kj_z = np.conj(kj_z)

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), self.d_list[-1], self.z_step)

        # Evaluate E field
        E = Wmp1 * exp(1j * kj_z * z) + Xmp1 * exp(-1j * kj_z * z)

        return {'z': z, 'E': E}

    def layer_E_field(self, layer):
        # self._simulation_test()
        d_list = self.d_list
        d_cumsum = self.d_cumsum
        n = self.n_list

        # Wave vector components in layer
        qj = self.q(n[layer], n[0], self.th)
        kj_z = (2 * pi * qj) / self.lam_vac

        # Time reversed
        kj_z = np.conj(kj_z)

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), d_list[layer], self.z_step)

        # calculate the transfer matrices
        S = self.s_mat()
        S_prime = self.s_primed_mat(layer)
        det_S_prime = self.matrix_2x2_determinant(S_prime)

        # Layer coefficients (time reversal BCs)
        X0 = 1 / n[0]
        rR = S[0, 1] / S[1, 1]
        Wj = X[0] * (rR * S_prime[1, 1] - S_prime[0, 1]) / det_S_prime
        Xj = X[0] * (S_prime[0, 0] - rR * S_prime[1, 0]) / det_S_prime

        # Evaluate E field
        dj = d_list[layer]
        zj = d_cumsum[layer]
        E = Wj * exp(1j * kj_z * (z - zj - dj/2)) + Xj * exp(-1j * kj_z * (z - zj - dj/2))

        return {'z': z, 'E': E}

    def structure_E_field(self):
        # x positions to evaluate E field at over entire structure
        z_pos = np.arange((self.z_step / 2.0), sum(self.d_list), self.z_step)
        # get x_mat - specifies what layer the corresponding point in x_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z_pos)
        comp2 = np.transpose(np.kron(np.ones((len(z_pos), 1)), self.d_cumsum))
        z_mat = sum(comp1 > comp2, 0)

        E = np.zeros(len(z_pos), dtype=complex)

        # Radiative lower outgoiong modes
        n = self.n_list
        S = self.s_mat()

        # # Calculate lower cladding E field
        z_indices = np.where(z_mat == 0)
        X0 = 1 / n[0]
        W0 = (S[0, 1] / S[1, 1]) * X0
        E[z_indices] = self.lower_cladding_E_field(W0, X0)['E']

        # # Calculate internal E field
        # for layer in range(1, self.num_layers - 1):
        #     # Calculate z indices inside structure for the layer
        #     z_indices = np.where(z_mat == layer)
        #     E_layer = self.layer_E_field(layer=layer)['E']
        #     E[z_indices] = E_layer

        # # Calculate upper cladding E field
        z_indices = np.where(z_mat == self.num_layers - 1)
        Wmp1 = 0
        Xmp1 = (1/S[1, 1]) * X0
        E[z_indices] = self.upper_cladding_E_field(Wmp1, Xmp1)['E']

        # Radiative upper outgoiong modes
        # # Calculate upper cladding E field
        # Wmp1 = 1 / n[0]
        # Xmp1 = -(S[1, 0] / S[1, 1]) * Wmp1
        # E[z_indices] = self.upper_cladding_E_field(Wmp1, Xmp1)['E']

        E_square = abs(E[:]) ** 2
        return {'z': z_pos, 'E': E, 'E_square': E_square}

def spe():
    st = LifetimeTmm()

    # st.add_layer(0, 3.48)
    st.add_layer(2000, 3.48)
    st.add_layer(2000, 1)
    # st.add_layer(0, 1)

    st.set_wavelength(1550)
    st.set_polarization('s')

    result = st.spe_structure()
    y = result['spe']
    z = result['z']

    plt.figure()
    plt.plot(z, y)
    plt.axhline(y=1, linestyle='--', color='k')
    plt.xlabel('Position in layer (nm)')
    plt.ylabel('Purcell Factor')
    dsum = getattr(st, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, zmat in enumerate(dsum):
        plt.axvline(x=zmat, linestyle='-', color='r', lw=2)
    plt.show()


def test():
    st = LifetimeTmm()
    # st.add_layer(0, 3.48)
    st.add_layer(2000, 3.48)
    st.add_layer(2000, 1)
    # st.add_layer(0, 1)
    st.set_wavelength(1550)
    st.set_polarization('s')

    plt.figure()
    for th in [0, 10, 70]:
        st.set_angle(th, units='degrees')
        y = st.structure_E_field()['E_square']
        plt.plot(y, label=th)
    plt.legend()

    # Plot boundaries
    dsum = getattr(st, 'd_cumsum')
    plt.axhline(y=1, linestyle='--', color='k')
    for i, zmat in enumerate(dsum):
        plt.axvline(x=zmat, linestyle='-', color='r', lw=2)
    plt.xlabel('Position in Device (nm)')
    plt.ylabel('Normalized |E|$^2$Intensity')
    plt.show()


if __name__ == "__main__":
    # spe()
    test()
