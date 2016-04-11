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

    def layer_E_field(self, layer, time_reversal=False):
        self._simulation_test()
        d_list = self.d_list
        n = self.n_list

        # Wave vector components in layer
        qj = self.q(n[layer], n[0], self.th)
        kj_z = (2 * pi * qj) / self.lam_vac

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), d_list[layer], self.z_step)

        # calculate the transfer matrices
        S = self.s_mat()
        S_prime = self.s_primed_mat(layer)
        det_S_prime = self.matrix_2x2_determinant(S_prime)
        if not time_reversal:
            # In units of i_l (lower incoming wave amplitude)
            rR = S[1, 0] / S[0, 0]
            E_plus = (S_prime[1, 1] - rR * S_prime[0, 1]) / det_S_prime
            E_minus = (rR * S_prime[0, 0] - S_prime[1, 0]) / det_S_prime
            E = E_plus * exp(1j * kj_z * z) + E_minus * exp(-1j * kj_z * z)
        else:  # Time reversal
            # In units of X_0 (lower outgoing wave amplitude)
            kj_z = np.conj(kj_z)
            rR = S[0, 1] / S[1, 1]
            E_plus = (rR*S_prime[1, 1] - S_prime[0, 1]) / det_S_prime
            E_minus = (S_prime[0, 0] - rR*S_prime[1, 0]) / det_S_prime
            E = E_plus * exp(1j * kj_z * z) + E_minus * exp(-1j * kj_z * z)

        E_square = abs(E[:])**2
        E_avg = sum(E_square) / (self.z_step * d_list[layer])
        return {'z': z, 'E': E, 'E_square': E_square, 'E_avg': E_avg}

    def structure_E_field(self, time_reversal=False):
        # x positions to evaluate E field at over entire structure
        z_pos = np.arange((self.z_step / 2.0), sum(self.d_list), self.z_step)
        # get x_mat - specifies what layer the corresponding point in x_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z_pos)
        comp2 = np.transpose(np.kron(np.ones((len(z_pos), 1)), self.d_cumsum))
        z_mat = sum(comp1 > comp2, 0)

        E = np.zeros(len(z_pos), dtype=complex)
        for layer in range(1, self.num_layers-1):
            # Calculate z indices inside structure for the layer
            z_indices = np.where(z_mat == layer)
            E_layer = self.layer_E_field(layer=layer, time_reversal=time_reversal)['E']
            E[z_indices] = E_layer
        E_square = abs(E[:])**2
        return {'z': z_pos, 'E': E, 'E_square': E_square}

    def spe_layer(self, layer, emission_direction='Lower'):

        assert self.n_list[0] >= self.n_list[ -1], \
            'Refractive index of lower cladding must be larger than the upper cladding'

        n_layer = self.n_list[layer].real

        if emission_direction == 'Upper':  # Flip structure
            self.flip()
            layer = self.num_layers - layer - 1

        resolution = 2**11 + 1
        th_emission, dth = np.linspace(0, pi/2, resolution, endpoint=False, retstep=True)
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        E_square_theta = np.zeros((resolution, len(z)), dtype=float)

        # Params for tqdm progress bar
        kwargs = {
            'total': resolution,
            'unit': 'theta modes',
            'unit_scale': True,
        }

        for i, th in tqdm(enumerate(th_emission), **kwargs):
            self.set_angle(th)

            # Calculate E field within layer
            E = self.layer_E_field(layer=layer, time_reversal=True)['E']
            # Normalise for X_0 = 1/ n_0
            E /= self.n_list[0].real

            if emission_direction == 'Upper':  # Flip E field to forward structure orientation
                E = E[::-1]

            E_square = abs(E[:]) ** 2
            E_square_theta[i, :] = E_square * sin(th)

        if emission_direction == 'Upper':  # Flip back to forward structure
            self.flip()

        # Evaluate spontaneous emission rate (axis=0 integrates over each columns)
        spe = integrate.romb(E_square_theta, dx=dth, axis=0) * n_layer**3
        # Normalise to vacuum emission rate of a randomly orientated dipole
        spe *= (3/8)
        return {'z': z, 'spe': spe}

    def spe_structure(self):
        # x positions to evaluate E field at over entire structure
        z_pos = np.arange((self.z_step / 2.0), sum(self.d_list), self.z_step)
        # get x_mat - specifies what layer the corresponding point in x_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z_pos)
        comp2 = np.transpose(np.kron(np.ones((len(z_pos), 1)), self.d_cumsum))
        x_mat = sum(comp1 > comp2, 0)
        # Evaluate spontaneous emission rate for each medium inside cladding layers
        spe = np.zeros(len(z_pos), dtype=float)
        for layer in range(1, self.num_layers-1):
            # Calculate x indices inside structure for the layer
            x_indices = np.where(x_mat == layer)
            spe[x_indices] += self.spe_layer(layer=layer, emission_direction='Lower')['spe']
            spe[x_indices] += self.spe_layer(layer=layer, emission_direction='Upper')['spe']
        return {'z': z_pos, 'spe': spe}