import scipy as sp
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from numpy import pi, exp, sin, sqrt, sum
from numpy.linalg import det
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
        """ Set the vacuum wavelength to be simulated.
        Note to ensure that dimensions are consistent with layer thicknesses."""
        self.lam_vac = lam_vac

    def set_polarization(self, pol):
        """ Set the mode polarisation to be simulated. 's' == TE and 'p' == TM """
        if pol not in ['s', 'p'] and self.th == 0:
            raise ValueError("Polarisation must be 's' or 'p' when angle of incidence is"
                             " not 0$\degree$s")
        self.pol = pol

    def set_angle(self, th, units='radians'):
        """ Set the angle of the simulated mode."""
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
        return sqrt(nj**2 - (n0*sin(th))**2)

    def I_mat(self, nj, nk):
        n0 = self.n_list[0]
        qj = self.q(nj, n0.real, self.th)
        qk = self.q(nk, n0.real, self.th)
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

    def S_primed_mat(self, layer):
        n = self.n_list
        # Calculate S_Prime
        S_prime = self.I_mat(n[0], n[1])
        for v in range(1, layer):
            mL = self.L_mat(n[v], self.d_list[v])
            mI = self.I_mat(n[v], n[v + 1])
            S_prime = S_prime @ mL @ mI
        return S_prime

    def S_dprimed_mat(self, layer):
        d_list = self.d_list
        n = self.n_list
        # Calculate S_dPrime (doubled prime)
        S_dprime = self.I_mat(n[layer], n[layer + 1])
        for v in range(layer + 1, self.num_layers - 1):
            mL = self.L_mat(n[v], d_list[v])
            mI = self.I_mat(n[v], n[v + 1])
            S_dprime = S_dprime @ mL @ mI
        return S_dprime

    def E_field_layer(self, layer, radiative='Lower'):
        # self._simulation_test()

        # Wave vector components in layer
        qj = self.q(self.n_list[layer], self.n_list[0], self.th)
        k_z = (2 * pi * qj) / self.lam_vac

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)

        # Evaluate fwd and bkwd coefficients for the layer
        S = self.s_mat()
        if radiative == 'Lower':
            X_0 = 1 / self.n_list[0]
            if layer == 0:  # Evaluate lower cladding
                W = (S[0, 1] / S[1, 1]) * X_0
                X = X_0
                # Note W_0 and X_0 are defined at cladding-layer boundary
                z = -z[::-1]
            elif layer == self.num_layers - 1:  # Evaluate upper cladding
                W = 0
                X = (1 / S[1, 1]) * X_0
            else:  # Evaluate internal layer electric field
                # calculate the total and partial transfer matrices
                S_prime = self.S_primed_mat(layer)
                rR = S[0, 1] / S[1, 1]
                W = X_0 * (rR * S_prime[1, 1] - S_prime[0, 1]) / det(S_prime)
                X = X_0 * (S_prime[0, 0] - rR * S_prime[1, 0]) / det(S_prime)

        elif radiative == 'Upper':
            W_mp1 = 1 / self.n_list[-1]
            if layer == 0:  # Evaluate lower cladding
                W = (det(S) / S[1, 1]) * W_mp1
                X = 0
                # Note W_0 and X_0 are defined at cladding-layer boundary
                z = -z[::-1]
            elif layer == self.num_layers - 1:  # Evaluate upper cladding
                W = W_mp1
                X = (- S[1, 0] / S[1, 1]) * W_mp1
            else:  # Evaluate internal layer electric field
                # calculate the total and partial transfer matrices
                S_prime = self.S_primed_mat(layer)
                W_0 = det(S) * W_mp1 / S[1, 1]
                W = W_0 * S_prime[1, 1] / det(S_prime)
                X = - W_0 * S_prime[1, 0] / det(S_prime)

        # Invoke time reversal for propagating waves
        z = -z
        if self.n_list[-1] <= self.n_list[layer] * sin(self.th) <= self.n_list[0] and layer == self.num_layers - 1:
            # Then partially radiative mode (propagating lower and evanescent in upper
            print('Partially Radiative %f' % k_z)
            # k_z = np.conj(k_z)

        # Evaluate E field (normal fwd in time relation)
        E = W * exp(1j * k_z * z) + X * exp(-1j * k_z * z)

        E_square = abs(E[:]) ** 2
        return {'z': z, 'E': E, 'E_square': E_square}

    def spe_layer(self, layer):
        assert self.n_list[0] >= self.n_list[-1], \
            'Refractive index of lower cladding must be larger than the upper cladding'

        # z positions in layer to evaluate
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)

        resolution = 2 ** 11 + 1
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
            E_square_theta[i, :] += self.E_field_layer(layer=layer, radiative='Lower')['E_square'] * sin(theta)
            # E_square_theta[i, :] += self.E_field_layer(layer=layer, radiative='Upper')['E_square'] * sin(theta)

        # Evaluate spontaneous emission rate (axis=0 integrates all rows, containing thetas, over each columns, z)
        spe = integrate.romb(E_square_theta, dx=dth, axis=0) * self.n_list[layer].real ** 3
        # Normalise to vacuum emission rate of a randomly orientated dipole
        spe *= 3 / 8

        return {'z': z, 'spe': spe}

    def spe_structure(self):
        # z positions to evaluate E field at over entire structure
        z_pos = np.arange((self.z_step / 2.0), self.d_cumsum[-1], self.z_step)

        # get z_mat - specifies what layer the corresponding point in z_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z_pos)
        comp2 = np.transpose(np.kron(np.ones((len(z_pos), 1)), self.d_cumsum))
        z_mat = sum(comp1 > comp2, 0)

        spe = np.zeros(len(z_pos), dtype=float)
        for layer in range(self.num_layers):
            if layer == 0:
                print('Evaluating lower cladding...')
            elif layer == self.num_layers - 1:
                print('Evaluating upper cladding...')
            else:
                print('Evaluating layer %d...' % layer)
            # Calculate z indices inside structure for the layer
            ind = np.where(z_mat == layer)

            # Calculate TE modes
            self.set_polarization('s')
            spe[ind] = self.spe_layer(layer)['spe']

        return {'z': z_pos, 'spe': spe}


def calculation():
    # Create structure
    st = LifetimeTmm()
    # st.add_layer(2000, 1)
    st.add_layer(2000, 3.48)
    st.add_layer(2000, 1)

    # Set light info
    st.set_wavelength(1550)

    # Get results
    result = st.spe_structure()
    z = result['z']
    fp = result['spe']

    # Plot
    plt.figure()
    plt.plot(z, fp)
    plt.axhline(y=1, linestyle='--', color='k')
    plt.xlabel('Position in layer (nm)')
    plt.ylabel('Purcell Factor')
    plt.axhline(y=1, linestyle='--', color='k')
    # Plot layer boundaries
    for z_j in getattr(st, 'd_cumsum'):
        plt.axvline(x=z_j, color='r', lw=2)
    plt.show()


if __name__ == "__main__":
    calculation()
