import numpy as np
import scipy.integrate as integrate

from tqdm import *
from numpy import pi, sin, sum
from numpy.linalg import det
from lifetmm.Methods.TransferMatrix import TransferMatrix


class LifetimeTmm(TransferMatrix):
    # def __init__(self):
    #     super().__init__()
    #     self.time_rev = False

    def wave_vector(self, layer):
        # if self.radiative == 'Lower':
        #     n0 = self.n_list[0].real
        # else:  # self.radiative =='Upper'
        #     n0 = self.n_list[-1].real
        n0 = self.n_list[layer]

        k = 2 * pi * n0 / self.lam_vac
        k_11 = k * sin(self.th)
        qj = self.q(layer)
        k_z = (2 * pi * qj) / self.lam_vac
        return k, k_z, k_11

    def time_fwd_coeff(self, layer):
        # Evaluate fwd and bkwd coefficients in units of incoming wave amplitude
        S = self.s_mat()
        if self.radiative == 'Lower':
            rR = S[1, 0] / S[0, 0]
            if layer == 0:  # Evaluate lower cladding
                E_plus = 1
                E_minus = rR
            elif layer == self.num_layers - 1:  # Evaluate upper cladding
                E_plus = 1 / S[0, 0]
                E_minus = 0
            else:  # Evaluate internal layer electric field
                S_prime = self.s_primed_mat(layer)
                E_plus = (S_prime[1, 1] - rR * S_prime[0, 1]) / det(S_prime)
                E_minus = (rR * S_prime[0, 0] - S_prime[1, 0]) / det(S_prime)
        else:  # self.radiative == 'Upper':
            if layer == 0:  # Evaluate lower cladding
                E_plus = 0
                E_minus = det(S)/S[0, 0]
            elif layer == self.num_layers - 1:  # Evaluate upper cladding
                E_plus = - S[0, 1] / S[0, 0]
                E_minus = 1
            else:  # Evaluate internal layer electric field
                S_prime = self.s_primed_mat(layer)
                E_plus = -(S_prime[0, 1]/det(S_prime)) * (det(S)/S[0, 0])
                E_minus = (S_prime[0, 0]/det(S_prime)) * (det(S)/S[0, 0])
        return E_plus, E_minus

    def time_rev_coeff(self, layer):
        # Evaluate fwd and bkwd coefficients in units of outgoing wave amplitude
        S = self.s_mat()
        if self.radiative == 'Lower':
            if layer == 0:  # Evaluate lower cladding
                E_plus = S[0, 1] / S[1, 1]
                E_minus = 1
            elif layer == self.num_layers - 1:  # Evaluate upper cladding
                E_plus = 0
                E_minus = 1 / S[1, 1]
            else:  # Evaluate internal layer electric field
                # calculate the total and partial transfer matrices
                S_prime = self.s_primed_mat(layer)
                rR = S[0, 1] / S[1, 1]
                E_plus = (rR * S_prime[1, 1] - S_prime[0, 1]) / det(S_prime)
                E_minus = (S_prime[0, 0] - rR * S_prime[1, 0]) / det(S_prime)
        else:  # self.radiative == 'Upper':
            if layer == 0:  # Evaluate lower cladding
                E_plus = det(S) / S[1, 1]
                E_minus = 0
            elif layer == self.num_layers - 1:  # Evaluate upper cladding
                E_plus = 1
                E_minus = - S[1, 0] / S[1, 1]
            else:  # Evaluate internal layer electric field
                # calculate the total and partial transfer matrices
                S_prime = self.s_primed_mat(layer)
                E_plus_0 = det(S) / S[1, 1]
                E_plus = E_plus_0 * S_prime[1, 1] / det(S_prime)
                E_minus = - E_plus_0 * S_prime[1, 0] / det(S_prime)
        return E_plus, E_minus

    def spe_layer(self, layer):
        # spe rate should be calculated backward in time (see paper)... or does it?
        self.time_rev = False

        assert self.n_list[0] >= self.n_list[-1], \
            'Refractive index of lower cladding must be larger than the upper cladding'

        # z positions in layer to evaluate
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)

        resolution = 2 ** 8 + 1
        theta_input, dth = np.linspace(0, pi / 2, num=resolution, endpoint=False, retstep=True)
        E_square_theta = np.zeros((len(theta_input), len(z)), dtype=float)

        # Params for tqdm progress bar
        kwargs = {
            'total': resolution,
            'unit': ' theta',
            'unit_scale': True,
        }

        for i, theta in tqdm(enumerate(theta_input), **kwargs):
            self.set_angle(theta)

            # Calculate E field within layer
            E = self.layer_E_field(layer)['E']

            # Normalise for outgoing wave medium refractive index - only TE
            if self.pol in ['s', 'TE']:
                if self.radiative == 'Lower':
                    E /= self.n_list[0].real
                else:  # radiative == 'Upper'
                    E /= self.n_list[-1].real

            # Wave vector components in layer
            k, k_z, k_11 = self.wave_vector(layer)

            # TODO: TM Mode check
            if self.pol in ['p', 'TE']:
                if self.dipole == 'Vertical':
                    E *= k_11
                else:  # self.dipole == 'Horizontal'
                    E *= k_z

            E_square_theta[i, :] += abs(E)**2 * sin(theta)

        # Evaluate spontaneous emission rate
        # (axis=0 integrates all rows, containing thetas, over each columns, z)
        spe = integrate.romb(E_square_theta, dx=dth, axis=0)

        if self.radiative == 'Lower':
            spe *= self.n_list[0].real ** 3
        else:  # radiative == 'Upper'
            spe *= self.n_list[-1].real ** 3

        # Normalise to vacuum emission rate of a randomly orientated dipole
        spe *= 3/8
        # TODO: TM Mode check
        if self.pol in ['p', 'TE']:
            spe *= ((self.lam_vac*1E-9)**2) / (4 * pi**2 * self.n_list[layer].real ** 4)

        return {'z': z, 'spe': spe}

    def spe_structure(self):
        """ Return the spontaneous emission rate vs z of the structure for each dipole orientation. """
        # z positions to evaluate E field at over entire structure
        z_pos = np.arange((self.z_step / 2.0), self.d_cumsum[-1], self.z_step)

        # get z_mat - specifies what layer the corresponding point in z_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z_pos)
        comp2 = np.transpose(np.kron(np.ones((len(z_pos), 1)), self.d_cumsum))
        z_mat = sum(comp1 > comp2, 0)

        spe_TE_Lower = np.zeros(len(z_pos), dtype=float)
        spe_TE_Upper = np.zeros(len(z_pos), dtype=float)
        spe_TM_Lower_h = np.zeros(len(z_pos), dtype=float)
        spe_TM_Upper_h = np.zeros(len(z_pos), dtype=float)
        spe_TM_Lower_v = np.zeros(len(z_pos), dtype=float)
        spe_TM_Upper_v = np.zeros(len(z_pos), dtype=float)
        for layer in range(self.num_layers):
            if layer == 0:
                print('\nEvaluating lower cladding...')
            elif layer == self.num_layers - 1:
                print('\nEvaluating upper cladding...')
            else:
                print('\nEvaluating internal layer: %d...' % layer)

            ind = np.where(z_mat == layer)

            # Calculate TE modes
            self.set_polarization('s')
            self.radiative = 'Lower'
            spe_TE_Lower[ind] += self.spe_layer(layer)['spe']
            self.radiative = 'Upper'
            spe_TE_Upper[ind] += self.spe_layer(layer)['spe']

            # Calculate TM modes
            self.set_polarization('p')

            self.dipole = 'Horizontal'
            self.radiative = 'Lower'
            spe_TM_Lower_h[ind] += self.spe_layer(layer)['spe']
            self.radiative = 'Upper'
            spe_TM_Upper_h[ind] += self.spe_layer(layer)['spe']

            self.dipole = 'Vertical'
            self.radiative = 'Lower'
            spe_TM_Lower_v[ind] += self.spe_layer(layer)['spe']
            self.radiative = 'Upper'
            spe_TM_Upper_v[ind] += self.spe_layer(layer)['spe']

        spe_TE = spe_TE_Upper + spe_TE_Lower
        spe_TM_h = spe_TM_Upper_h + spe_TM_Lower_h
        spe_TM_v = spe_TM_Upper_v + spe_TM_Lower_v
        spe = spe_TE_Lower + spe_TE_Upper + spe_TM_Upper_h + spe_TM_Upper_h + spe_TM_Upper_v + spe_TM_Upper_v
        return {'z': z_pos, 'spe': spe, 'spe_TE': spe_TE, 'spe_TM_h': spe_TM_h, 'spe_TM_v': spe_TM_v}
