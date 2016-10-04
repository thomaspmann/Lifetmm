import numpy as np
import scipy as sp
import scipy.integrate as integrate
from tqdm import *
from numpy import pi, sin, sum, sqrt, exp
from numpy.linalg import det
# from lifetmm.Methods.TransferMatrix import TransferMatrix


class TransferMatrix:
    def __init__(self):
        self.d_list = np.array([], dtype=float)
        self.n_list = np.array([], dtype=complex)
        self.d_cumsum = np.array([], dtype=float)
        self.z_step = 1
        self.lam_vac = 0
        self.num_layers = 0
        self.pol = 'u'
        self.th = 0
        self.radiative = 'Lower'

    def add_layer(self, d, n):
        self.d_list = np.append(self.d_list, d)
        self.n_list = np.append(self.n_list, n)
        self.d_cumsum = np.cumsum(self.d_list)
        self.num_layers = np.size(self.d_list)

    def set_wavelength(self, lam_vac):
        """ Set the vacuum wavelength to be simulated.
        Note to ensure that dimensions are consistent with layer thicknesses."""
        self.lam_vac = lam_vac

    def set_polarization(self, pol):
        """ Set the mode polarisation to be simulated. 's' == TE and 'p' == TM """
        if pol not in ['s', 'p', 'TE', 'TM'] and self.th != 0:
            raise ValueError("Polarisation must be defined when angle of incidence is"
                             " not 0$\degree$s")
        self.pol = pol

    def set_angle(self, th, units='radians'):
        """ Set the angle of the simulated mode."""
        if hasattr(th, 'size') and th.size > 1:
            raise ValueError('This function is not vectorized; you need to run one '
                             'calculation for each angle at a time')
        if units == 'radians':
            assert 0 <= th < pi / 2, 'The light is not incident on the structure. ' \
                                     'Check input theta satisfies -pi/2 <= theta < pi/2'
            self.th = th
        elif units == 'degrees':
            assert 0 <= th < 90, 'The light is not incident on the structure. ' \
                                 'Check input theta satisfies -90 <= theta < 90'
            self.th = th * (pi / 180)
        else:
            raise ValueError('Units of angle not recognised. Please enter \'radians\' or \'degrees\'.')

    def wave_vector(self, layer):
        # Free space wave vector
        if self.radiative == 'Lower':
            n0 = self.n_list[0].real
        else:  # self.radiative =='Upper'
            n0 = self.n_list[-1].real
        k0 = 2 * pi * n0 / self.lam_vac

        # Layer wave vector and components
        n = self.n_list[layer].real
        k = 2 * pi * n / self.lam_vac
        k_11 = k0 * sin(self.th)  # Note th needs to be in same layer as k0
        q = sp.sqrt(k**2 - k_11**2)
        # q = (2 * pi * self.q(layer)) / self.lam_vac  # (equivalent to above)
        return k, q, k_11

    def q(self, j):
        if self.radiative == 'Lower':
            n0 = self.n_list[0].real
        else:  # self.radiative =='Upper'
            n0 = self.n_list[-1].real
        nj = self.n_list[j]
        return sqrt(nj**2 - (n0*sin(self.th))**2)

    def I_mat(self, j, k):
        """ Returns the interference matrix between layers j and k."""
        qj = self.q(j)
        qk = self.q(k)
        nj = self.n_list[j]
        nk = self.n_list[k]
        if self.pol in ['p', 'TM']:
            r = (qj * nk ** 2 - qk * nj ** 2) / (qj * nk ** 2 + qk * nj ** 2)
            t = (2 * nj * nk * qj) / (qj * nk ** 2 + qk * nj ** 2)
            # Convert t_E to t_H
            t *= nk/nj
        else:  # self.pol in ['s', 'TE', 'u']:
            r = (qj - qk) / (qj + qk)
            t = (2 * qj) / (qj + qk)
        assert t != 0, ValueError('Transmission is zero, cannot evaluate I_mat.')
        return (1 / t) * np.array([[1, r], [r, 1]], dtype=complex)

    def L_mat(self, j):
        """ Returns the propagation matrix for layer j."""
        qj = self.q(j)
        dj = self.d_list[j]
        eps = (2*pi*qj) / self.lam_vac
        return np.array([[exp(-1j*eps*dj), 0], [0, exp(1j*eps*dj)]], dtype=complex)

    def s_mat(self):
        """ Returns the total system transfer matrix S."""
        S = self.I_mat(0, 1)
        for layer in range(1, self.num_layers - 1):
            mL = self.L_mat(layer)
            mI = self.I_mat(layer, layer + 1)
            S = S @ mL @ mI
        return S

    def s_primed_mat(self, layer):
        """ Returns the partial system transfer matrix S_prime."""
        S_prime = self.I_mat(0, 1)
        for j in range(1, layer):
            mL = self.L_mat(j)
            mI = self.I_mat(j, j + 1)
            S_prime = S_prime @ mL @ mI
        return S_prime

    def s_dprimed_mat(self, layer):
        """ Returns the partial system transfer matrix S_dprime (doubled prime)."""
        S_dprime = self.I_mat(layer, layer + 1)
        for j in range(layer + 1, self.num_layers - 1):
            mL = self.L_mat(j)
            mI = self.I_mat(j, j + 1)
            S_dprime = S_dprime @ mL @ mI
        return S_dprime

    def amplitude_E(self, layer):
        # Evaluate layer fwd and bkwd coefficients in units of incoming wave amplitude
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

    def _simulation_test(self):
        if type(self.z_step) != int:
            raise ValueError('z_step must be an integer. Reduce SI unit'
                             'inputs for thicknesses and wavelengths for greater resolution ')

    def get_layer_boundaries(self):
        """ Return layer boundary zs assuming that the lower cladding boundary is at z=0. """
        return self.d_cumsum


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
