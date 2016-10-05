import numpy as np
import scipy as sp
import scipy.integrate as integrate
from tqdm import *
from numpy import pi, sin, sum, sqrt, exp
from numpy.linalg import det
from lifetmm.Methods.TransferMatrix import TransferMatrix


class LifetimeTmm(TransferMatrix):
    def spe_layer(self, layer):
        assert self.n_list[0] >= self.n_list[-1], \
            'Refractive index of lower cladding must be larger than the upper cladding'

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        if layer == 0:
            # Note E_plus and E_minus are defined at cladding-layer boundary
            z = -z[::-1]

        resolution = 2 ** 8 + 1
        theta_input, dth = np.linspace(0, pi / 2, num=resolution, endpoint=False, retstep=True)
        E_TE_square_theta = np.zeros((len(theta_input), len(z)), dtype=float)
        E_TM_p_square_theta = np.zeros((len(theta_input), len(z)), dtype=float)
        E_TM_s_square_theta = np.zeros((len(theta_input), len(z)), dtype=float)

        # Params for tqdm progress bar
        kwargs = {
            'total': resolution,
            'unit': ' theta',
            'unit_scale': True,
        }
        for i, theta in tqdm(enumerate(theta_input), **kwargs):
            # Set the angle to be evaluated
            self.set_angle(theta)

            # Wave vector components in layer (q, k_11 are angle dependent)
            k, q, k_11 = self.wave_vector(layer)

            # !* TE modes *!
            # Calculate E field within layer
            self.set_polarization('TE')
            self.field = 'E'
            # E field coefficients in terms of E_0^+
            E_plus, E_minus = self.amplitude_E(layer)
            E_TE = E_plus * exp(1j * q * z) + E_minus * exp(-1j * q * z)
            # Orthonormality: Normalise outgoing TE wave to medium refractive index
            if self.radiative == 'Lower':
                E_TE /= self.n_list[0].real
            else:  # radiative == 'Upper'
                E_TE /= self.n_list[-1].real

            # !* TM modes *!
            # Calculate H field within layer
            self.set_polarization('TM')
            self.field = 'H'
            # E field coefficients in terms of E_0^+
            H_plus, H_minus = self.amplitude_E(layer)
            # Calculate the electric field component perpendicular to the interface
            E_TM_s = k_11*(H_plus * exp(1j * q * z) + H_minus * exp(-1j * q * z))
            # Calculate the electric field component parallel to the interface
            E_TM_p = q*(H_plus * exp(1j * q * z) - H_minus * exp(-1j * q * z))

            # Take the squares of all E field components and add weighting
            E_TE_square_theta[i, :] += abs(E_TE) ** 2 * sin(theta)
            E_TM_s_square_theta[i, :] += abs(E_TM_s) ** 2 * sin(theta)
            E_TM_p_square_theta[i, :] += abs(E_TM_p) ** 2 * sin(theta)

        # Evaluate spontaneous emission rate for each z (columns) over all thetas (rows)
        spe_TE = integrate.romb(E_TE_square_theta, dx=dth, axis=0)
        spe_TM_p = integrate.romb(E_TM_p_square_theta, dx=dth, axis=0)
        spe_TM_s = integrate.romb(E_TM_s_square_theta, dx=dth, axis=0)

        # Outgoing E mode refractive index weighting (just after summation over j=0,M+1)
        if self.radiative == 'Lower':
            for spe in [spe_TE, spe_TM_p, spe_TM_s]:
                spe *= self.n_list[0].real ** 3
        else:  # radiative == 'Upper'
            for spe in [spe_TE, spe_TM_p, spe_TM_s]:
                spe *= self.n_list[-1].real ** 3

        # Normalise emission rates to vacuum emission rate of a randomly orientated dipole
        # Wave vector in layer
        k, q, k_11 = self.wave_vector(layer)
        spe_TE *= 3/8
        nj = self.n_list[layer].real
        spe_TM_p *= 3/(8*(nj*k)**2)
        spe_TM_s *= 3/(4*(nj*k)**2)

        return {'z': z, 'spe_TE': spe_TE, 'spe_TM_s': spe_TM_s, 'spe_TM_p': spe_TM_p}

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
        spe_TM_p_Upper = np.zeros(len(z_pos), dtype=float)
        spe_TM_p_Lower = np.zeros(len(z_pos), dtype=float)
        spe_TM_s_Upper = np.zeros(len(z_pos), dtype=float)
        spe_TM_s_Lower = np.zeros(len(z_pos), dtype=float)
        for layer in range(self.num_layers):
            if layer == 0:
                print('Evaluating lower cladding...')
            elif layer == self.num_layers - 1:
                print('\nEvaluating upper cladding...')
            else:
                print('\nEvaluating internal layer: %d...' % layer)

            # Find indices corresponding to the layer we are evaluating
            ind = np.where(z_mat == layer)

            # Calculate lower radiative modes
            self.radiative = 'Lower'
            spe = self.spe_layer(layer)
            spe_TE_Lower[ind] += spe['spe_TE']
            spe_TM_s_Lower[ind] += spe['spe_TM_s']
            spe_TM_p_Lower[ind] += spe['spe_TM_p']
            # Calculate upper radiative modes
            self.radiative = 'Upper'
            spe = self.spe_layer(layer)
            spe_TE_Upper[ind] += spe['spe_TE']
            spe_TM_s_Upper[ind] += spe['spe_TM_s']
            spe_TM_p_Upper[ind] += spe['spe_TM_p']

        spe_TE = spe_TE_Upper + spe_TE_Lower
        spe_TM_s = spe_TM_s_Upper + spe_TM_s_Lower
        spe_TM_p = spe_TM_p_Upper + spe_TM_p_Lower

        return {'z': z_pos, 'spe_TE': spe_TE, 'spe_TM_s': spe_TM_s, 'spe_TM_p': spe_TM_p}
