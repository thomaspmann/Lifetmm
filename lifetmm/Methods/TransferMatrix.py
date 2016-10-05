import numpy as np
import scipy as sp
from numpy import pi, sqrt, sin, exp
from numpy.linalg import det


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
        self.field = 'E'

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
        # TODO: why is this required for mcGhee simulation but the above for spe???
        # Think its to do with when n is complex the above breaks down as k is defined with n.real
        # q = (2 * pi * self.q(layer)) / self.lam_vac  # (equivalent to above)?
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
        else:  # self.pol in ['s', 'TE', 'u']:
            r = (qj - qk) / (qj + qk)
            t = (2 * qj) / (qj + qk)
        if self.field == 'H':
            # Convert t_E to t_H
            t *= nk / nj
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

    def layer_E_field(self, layer):
        self._simulation_test()

        # Wave vector components in layer
        k, q, k_11 = self.wave_vector(layer)

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        if layer == 0:
            # Note E_plus and E_minus are defined at cladding-layer boundary
            z = -z[::-1]

        # E field in terms of E_0^+
        E_plus, E_minus = self.amplitude_E(layer)

        E = E_plus * exp(1j * q * z) + E_minus * exp(-1j * q * z)
        E_square = abs(E)**2

        if self.d_list[layer] != 0:
            E_avg = sum(E_square) / (self.z_step * self.d_list[layer])
        else:
            E_avg = np.nan
        return {'z': z, 'E': E, 'E_square': E_square, 'E_avg': E_avg}

    def structure_E_field(self):
        # x positions to evaluate E field at over entire structure
        z_pos = np.arange((self.z_step / 2.0), self.d_cumsum[-1], self.z_step)
        # get x_mat - specifies what layer the corresponding point in x_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z_pos)
        comp2 = np.transpose(np.kron(np.ones((len(z_pos), 1)), self.d_cumsum))
        z_mat = sum(comp1 > comp2, 0)
        # Initialise E array
        E = np.zeros(len(z_pos), dtype=complex)
        for layer in range(self.num_layers):
            # Calculate z indices inside structure for the layer
            z_indices = np.where(z_mat == layer)
            E_layer = self.layer_E_field(layer)['E']
            E[z_indices] = E_layer
        E_square = abs(E)**2
        return {'z': z_pos, 'E': E, 'E_square': E_square}

    def show_structure(self):
        """ Brings up a plot showing the structure."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from collections import OrderedDict

        # Shades to fill rectangles with based on refractive index
        alphas = abs(self.n_list) / max(abs(self.n_list))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, dx in enumerate(self.d_list[1:-1]):
            x = self.d_cumsum[i]
            layer_text = ('{0.real:.2f} + {0.imag:.2f}j'.format(self.n_list[i + 1]))
            p = patches.Rectangle(
                (x, 0.0),  # (x,y)
                dx,  # width
                1.0,  # height
                alpha=alphas[i + 1],
                linewidth=2,
                label=layer_text,
            )
            ax.add_patch(p)
        # Create legend without duplicate keys
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
        ax.set_xlim([0, self.d_cumsum[-1]])
        ax.set(xlabel=r'x', ylabel=r'A.U.')
        plt.show()

    def get_layer_boundaries(self):
        """ Return layer boundary zs assuming that the lower cladding boundary is at z=0. """
        return self.d_cumsum

    def calc_R_and_T(self):
        """ Return the reflection and transmission coefficients of the structure. """
        S = self.s_mat()
        R = abs(S[1, 0] / S[0, 0]) ** 2
        T = abs(1 / S[0, 0]) ** 2
        # note this is incorrect T: https://en.wikipedia.org/wiki/Fresnel_equations
        return R, T

    def calc_absorption(self):
        n = self.n_list
        # Absorption coefficient in 1/cm
        absorption = np.zeros(self.num_layers)
        for layer in range(1, self.num_layers):
            absorption[layer] = (4 * pi * n[layer].imag) / (self.lam_vac * 1.0e-7)
        return absorption

    def flip(self):
        """ Flip the structure front-to-back."""
        self.d_list = self.d_list[::-1]
        self.n_list = self.n_list[::-1]
        self.d_cumsum = np.cumsum(self.d_list)

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
        return sp.arcsin(np.real_if_close(n_1 * sin(th_1) / n_2))

    def _simulation_test(self):
        if type(self.z_step) != int:
            raise ValueError('z_step must be an integer. Reduce SI unit'
                             'inputs for thicknesses and wavelengths for greater resolution ')
