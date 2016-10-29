import numpy as np
import scipy as sp
from numpy import pi, sqrt, sin, exp
from numpy.linalg import det

from lifetmm.Methods.HelperFunctions import roots, snell


class TransferMatrix:
    def __init__(self):
        self.d_list = np.array([], dtype=float)
        self.n_list = np.array([], dtype=complex)
        self.d_cumsum = np.array([], dtype=float)
        self.num_layers = 0
        self.lam_vac = 0
        self.pol = ''
        # Default simulation parameters
        self.field = 'E'
        self.th = 0
        self.guided = False
        self.z_step = 1

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
        """ Set the mode polarisation to be simulated ('s' or 'TE' and 'p' or 'TM')
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

    def k(self, j):
        """ Calculate the wave vector magnitude in layer j. Alternatively if j ==- 1
        then we calculate the vacuum wave vector (k=omega/c).
        """
        if j == -1:
            n = 1
        else:
            n = self.n_list[j].real
        return 2 * pi * n / self.lam_vac

    def q(self, j):
        """ Normalised perpendicular wave-vector in layer j.
        """
        # Normalised wave-vector in layer
        nj = self.n_list[j]

        # Continuous across layers, so can evaluate from input theta
        # and medium for incoming wave (hence radiative mode)
        if not self.guided:
            n0 = self.n_list[0].real
            n_11 = n0*sin(self.th)
        else:
            # TODO: This accounts for when theta=0 to give beta/k=n1. Now need max beta/k=n2 (where n2 is guided mode)
            n_11 = self.beta
        return sqrt(nj**2 - n_11**2)

    def k_11(self):
        """ NOTE: This will not work for guided modes as there is no incident radiation (theta).
        """
        # For radiative modes
        k0 = self.k(0)
        k_11 = k0 * sin(self.th)  # Note th needs to be in same layer as k0
        return k_11

    def wave_vector(self, j):
        """ The wave vector magnitude and it's components perpendicular and parallel
        to the interface inside the layer calculated from the incident angle of wave.
        """
        # Layer wave vector and components
        k = self.k(j)
        k_11 = self.k_11()
        k_vac = self.k(-1)
        q = self.q(j)*k_vac
        # TODO: this method will only give real part of q as k() is derived from n.real
        # q = sp.sqrt(k**2 - k_11**2)
        return k, q, k_11

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
            # Convert transmission coefficient for E field to that of the H field.
            # Note that the reflection coefficient is the same as the medium does not change.
            t *= nk / nj
        if t == 0:
            # Can't evaluate I_mat when transmission t==0 as 1/t == inf
            t = np.nan
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
        for j in range(1, self.num_layers - 1):
            mL = self.L_mat(j)
            mI = self.I_mat(j, j + 1)
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
        # Transfer matrix of system
        S = self.S_mat()
        # Reflection for incoming wave incident of LHS of structure
        r = S[1, 0] / S[0, 0]
        # Evaluate lower cladding
        if layer == 0:
            A_plus = 1
            A_minus = r
        # Evaluate upper cladding
        elif layer == self.num_layers - 1:
            A_plus = 1 / S[0, 0]
            A_minus = 0
        # Evaluate field amplitudes in internal layers
        else:
            S_prime = self.S_primed_mat(layer)
            A_plus = (S_prime[1, 1] - r * S_prime[0, 1]) / det(S_prime)
            A_minus = (r * S_prime[0, 0] - S_prime[1, 0]) / det(S_prime)
        return A_plus, A_minus

    def layer_field(self, layer):
        """ Evaluate the field (E or H) as a function of z (depth) into the layer, j.
        A_plus is the forward component of the field (e.g. E_j^+)
        A_minus is the backward component of the field (e.g. E_j^-)
        """
        # Wave vector components in layer
        k, q, k_11 = self.wave_vector(layer)

        # z positions to evaluate field at at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        # Note A_plus and A_minus are defined at cladding-layer boundary so need to
        # propagate wave 'backwards' in the lower cladding by reversing z
        if layer == 0:
            z = -z[::-1]

        # A(z) field in terms of incident field amplitude (A_0^+)
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

    def s11_vs_beta_guided(self):
        """ Evaluate S_11=(1/t) as a function of beta (k_ll) in the guided regime.
        When S_11 = 0 the corresponding beta is a guided mode.
        """
        self.guided = True
        n = self.n_list.real

        beta = np.linspace(n[0], max(n), num=1000, endpoint=False)[1:]
        S_11 = np.array([])
        for b in beta:
            self.beta = b
            S = self.S_mat()
            S_11 = np.append(S_11, S[0, 0])
        self.guided = False
        return beta, S_11.real

    def find_guided_modes_beta(self):
        """ Evaluate beta at S_11=0 as a function of beta (k_ll) in the guided regime.
        """
        self.guided = True
        n = self.n_list.real

        def s_11(beta):
            # Evaluate transfer matrix element S_11 for a given beta
            self.beta = beta
            S = self.S_mat()
            s11 = S[0, 0].real
            # print(beta, s11)
            return s11

        root_list = roots(s_11, n[0], max(n))
        self.guided = False
        return root_list

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


    def calc_r_and_t(self):
        """ Return the complex reflection and transmission coefficients of the structure.
        """
        S = self.S_mat()
        r = S[1, 0] / S[0, 0]
        t = 1 / S[0, 0]
        return r, t

    def calc_R_and_T(self, correction=True):
        """ Return the reflectance and transmittance of the structure.
        """
        r, t = self.calc_r_and_t()
        R = abs(r) ** 2
        T = abs(t) ** 2
        if correction:
            # note correction for T due to beam expansion
            # https://en.wikipedia.org/wiki/Fresnel_equations
            n_1 = self.n_list[0].real
            n_2 = self.n_list[-1].real
            th_out = snell(n_1, n_2, self.th)
            rho = n_2 / n_1
            m = np.cos(th_out)/np.cos(self.th)
            T *= rho*m
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

