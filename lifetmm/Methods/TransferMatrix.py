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
        self.pol = ''
        self.th = 0
        self.field = 'E'

    def add_layer(self, d, n):
        """ Add layer of thickness d and refractive index n to the structure.
        """
        self.d_list = np.append(self.d_list, d)
        self.n_list = np.append(self.n_list, n)
        self.d_cumsum = np.cumsum(self.d_list)
        self.num_layers = np.size(self.d_list)

    def set_wavelength(self, lam_vac):
        """ Set the vacuum wavelength to be simulated.
        Note to ensure that dimensions must be consistent with layer thicknesses.
        """
        if hasattr(lam_vac, 'size') and lam_vac.size > 1:
            raise ValueError('This function is not vectorized; you need to run one '
                             'calculation for each wavelength at a time')
        self.lam_vac = lam_vac

    def set_polarization(self, pol):
        """ Set the mode polarisation to be simulated. 's' == TE and 'p' == TM
        """
        if pol not in ['s', 'p', 'TE', 'TM'] and self.th != 0:
            raise ValueError("Polarisation must be defined when angle of incidence is"
                             " not 0$\degree$s")
        self.pol = pol

    def set_field(self, field):
        """ Set the field to be evaluated. Either 'E' (default) or 'H' field.
        """
        if field not in ['E', 'H']:
            raise ValueError("The field must be either 'E' of 'H'.")
        self.field = field

    def set_angle(self, th, units='radians'):
        """ Set the angle of the simulated mode.
        """
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

    def set_z_step(self, step):
        """ Set the resolution in z of the simulation.
        """
        if type(step) != int:
            raise ValueError('z_step must be an integer. Reduce SI unit'
                             'inputs for thicknesses and wavelengths for greater resolution ')
        self.z_step = step

    def k0(self):
        """ Calculate the free space wave vector
        """
        n0 = self.n_list[0].real
        return 2 * pi * n0 / self.lam_vac

    def wave_vector(self, layer):
        """ The wave vector magnitude and it's components perpendicular and parallel
        to the interface inside the layer.
        """
        k0 = self.k0()

        # Layer wave vector and components
        n = self.n_list[layer].real
        k = 2 * pi * n / self.lam_vac
        k_11 = k0 * sin(self.th)  # Note th needs to be in same layer as k0
        # q = sp.sqrt(k**2 - k_11**2)
        # TODO: above breaks when n is complex the above breaks down as k is defined with n.real
        q = (2 * pi * self.q(layer)) / self.lam_vac
        return k, q, k_11

    def q(self, j):
        n0 = self.n_list[0].real
        nj = self.n_list[j]
        return sqrt(nj**2 - (n0*sin(self.th))**2)

    def I_mat(self, j, k):
        """ Returns the interference matrix between layers j and k.
        """
        qj = self.q(j)
        qk = self.q(k)
        nj = self.n_list[j]
        nk = self.n_list[k]
        # Evaluate reflection and transmission coefficients for E field
        if self.pol in ['p', 'TM']:
            r = (qj * nk ** 2 - qk * nj ** 2) / (qj * nk ** 2 + qk * nj ** 2)
            t = (2 * nj * nk * qj) / (qj * nk ** 2 + qk * nj ** 2)
        elif self.pol in ['s', 'TE']:
            r = (qj - qk) / (qj + qk)
            t = (2 * qj) / (qj + qk)
        else:
            raise ValueError('A polarisation for the field must be set.')
        if self.field == 'H':
            # Convert transmission coefficient for electric to that of the H field.
            # Note that the reflection coefficient is the same as the medium does not change.
            t *= nk / nj
        assert t != 0, ValueError('Transmission is zero, cannot evaluate I_mat.')
        return (1 / t) * np.array([[1, r], [r, 1]], dtype=complex)

    def L_mat(self, j):
        """ Returns the propagation matrix for layer j.
        """
        qj = self.q(j)
        dj = self.d_list[j]
        eps = (2*pi*qj) / self.lam_vac
        assert -1j*eps*dj < 25, \
            ValueError('L_matrix is unstable for such a large thickness with an exponentially growing mode.')
        return np.array([[exp(-1j*eps*dj), 0], [0, exp(1j*eps*dj)]], dtype=complex)

    def S_mat(self):
        """ Returns the total system transfer matrix S.
        """
        S = self.I_mat(0, 1)
        for layer in range(1, self.num_layers - 1):
            mL = self.L_mat(layer)
            mI = self.I_mat(layer, layer + 1)
            S = S @ mL @ mI
        return S

    def S_primed_mat(self, layer):
        """ Returns the partial system transfer matrix S_prime.
        """
        S_prime = self.I_mat(0, 1)
        for j in range(1, layer):
            mL = self.L_mat(j)
            mI = self.I_mat(j, j + 1)
            S_prime = S_prime @ mL @ mI
        return S_prime

    def S_dprimed_mat(self, layer):
        """ Returns the partial system transfer matrix S_dprime (doubled prime).
        """
        S_dprime = self.I_mat(layer, layer + 1)
        for j in range(layer + 1, self.num_layers - 1):
            mL = self.L_mat(j)
            mI = self.I_mat(j, j + 1)
            S_dprime = S_dprime @ mL @ mI
        return S_dprime

    def amplitude_coefficients(self, layer):
        """ Evaluate fwd and bkwd field amplitude coefficients (E or H) in a layer.
         Coefficients are in units of the fwd incoming wave amplitude.
        """
        S = self.S_mat()
        rR = S[1, 0] / S[0, 0]
        if layer == 0:  # Evaluate lower cladding
            A_plus = 1
            A_minus = rR
        elif layer == self.num_layers - 1:  # Evaluate upper cladding
            A_plus = 1 / S[0, 0]
            A_minus = 0
        else:  # Evaluate internal layer electric field
            S_prime = self.S_primed_mat(layer)
            A_plus = (S_prime[1, 1] - rR * S_prime[0, 1]) / det(S_prime)
            A_minus = (rR * S_prime[0, 0] - S_prime[1, 0]) / det(S_prime)
        return A_plus, A_minus

    def layer_field(self, layer):
        """ Evaluate the field (E or H) as a function of z (depth) into the layer, j.
        A_plus is the forward component of the field (E_j^+)
        A_minus is the backward component of the field (E_j^-)
        """
        # Wave vector components in layer
        k, q, k_11 = self.wave_vector(layer)

        # z positions to evaluate field at at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        if layer == 0:
            # Note A_plus and A_minus are defined at cladding-layer boundary
            z = -z[::-1]

        # A field in terms of E_0^+
        A_plus, A_minus = self.amplitude_coefficients(layer)

        A = A_plus * exp(1j * q * z) + A_minus * exp(-1j * q * z)
        A_squared = abs(A)**2

        if self.d_list[layer] != 0:
            A_avg = sum(A_squared) / (self.z_step * self.d_list[layer])
        else:
            A_avg = np.nan
        return {'z': z, 'A': A, 'A_squared': A_squared, 'A_avg': A_avg}

    def structure_field(self):
        """ Evaluate the field at all z positions within the structure.
        """
        z = np.arange((self.z_step / 2.0), self.d_cumsum[-1], self.z_step)
        # get z_mat - specifies what layer the corresponding point in z is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z)
        comp2 = np.transpose(np.kron(np.ones((len(z), 1)), self.d_cumsum))
        z_mat = sum(comp1 > comp2, 0)

        A = np.zeros(len(z), dtype=complex)
        for layer in range(self.num_layers):
            # Calculate z indices inside structure for the layer
            z_indices = np.where(z_mat == layer)
            A_layer = self.layer_field(layer)['A']
            A[z_indices] = A_layer
        A_squared = abs(A)**2
        return {'z': z, 'A': A, 'A_squared': A_squared}

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
        """ Return layer boundary zs assuming that the lower cladding boundary is at z=0.
        """
        return self.d_cumsum

    def get_structure_thickness(self):
        """ Return the structure thickness.
        """
        return self.d_cumsum[-1]

    def z_to_lambda(self, z, center=True):
        """ Convert z positions to units of wavelength (optional) from the centre.
        """
        if center:
            z -= self.get_structure_thickness()/2
        z /= self.lam_vac
        return z

    def calc_R_and_T(self):
        """ Return the reflection and transmission coefficients of the structure.
        """
        S = self.S_mat()
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
        """ Flip the structure front-to-back.
        """
        self.d_list = self.d_list[::-1]
        self.n_list = self.n_list[::-1]
        self.d_cumsum = np.cumsum(self.d_list)

    def info(self):
        """ Command line verbose feedback of the structure.
        """
        print('Simulation info.\n')

        print('Multi-layered Structure:')
        print('d\t\tn')
        for n, d in zip(self.n_list, self.d_list):
            print('{0:g}\t{1:g}'.format(d, n))
        print('\nFree space wavelength: {:g}\n'.format(self.lam_vac))

    @staticmethod
    def snell(n_1, n_2, th_1):
        """ Return angle theta in layer 2 with refractive index n_2, assuming
        it has angle th_1 in layer with refractive index n_1. Use Snell's law. Note
        that "angles" may be complex!!
        """
        # Important that the arcsin here is scipy.arcsin, not numpy.arcsin!! (They
        # give different results e.g. for arcsin(2).)
        # Use real_if_close because e.g. arcsin(2 + 1e-17j) is very different from
        # arcsin(2) due to branch cut
        return sp.arcsin(np.real_if_close(n_1 * sin(th_1) / n_2))
