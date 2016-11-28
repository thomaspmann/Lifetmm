import numpy as np
from numpy import pi, sqrt, sin, exp
from scipy.constants import c

from lifetmm.Methods.HelperFunctions import roots, snell, lambda2omega, det


class TransferMatrix:
    def __init__(self):
        self.d_list = np.array([], dtype=int)
        self.n_list = np.array([], dtype=complex)
        self.d_cumulative = np.array([], dtype=float)
        self.num_layers = 0
        self.lam_vac = 0
        self.k_vac = 0
        self.omega = 0
        # Default simulation parameters
        self.field = 'E'
        self.pol = 'TE'
        self.guided = False
        self.th = 0
        self.z_step = 1
        self.n_11 = 0  # Normalised parallel wave vector. Only defined and used for guiding modes.

    def add_layer(self, d, n):
        """
        Add layer of thickness d and refractive index n to the structure.
        """
        assert d >= 0, ValueError('Thickness must >= 0.')
        if not float(d).is_integer():
            print('WARNING: ROUNDING THICKNESS TO THE NEAREST INTEGER. '
                  'CONSIDER REDUCING SI UNITS FOR GREATER RESOLUTION.')
            d = round(d)
        self.d_list = np.append(self.d_list, d)
        self.n_list = np.append(self.n_list, n)
        # Recalculate structure info
        self.d_cumulative = np.cumsum(self.d_list)
        self.num_layers = np.size(self.d_list)

    def set_vacuum_wavelength(self, lam_vac):
        """
        Set the vacuum wavelength to be simulated.
        Note to ensure that dimensions must be consistent with layer thicknesses.
        """
        assert not hasattr(lam_vac, 'size'), ValueError('This function is not vectorized; you need to run one '
                                                        'calculation for each wavelength at a time')
        assert lam_vac > 0, ValueError('Wavelength must > 0.')
        if not float(lam_vac).is_integer():
            print('WARNING: ROUNDING WAVELENGTH TO THE NEAREST INTEGER. '
                  'CONSIDER REDUCING SI UNITS FOR GREATER RESOLUTION.')
            lam_vac = round(lam_vac)
        self.lam_vac = lam_vac
        self.k_vac = 2 * pi / lam_vac
        self.omega = lambda2omega(lam_vac)

    def set_z_step(self, step):
        """
        Set the resolution in z (perpendicular to the multilayer) of the simulation.
        Default is set to 1 if not called explicitly
        """
        if not float(step).is_integer():
            print('WARNING: ROUNDING STEP TO THE NEAREST INTEGER. '
                  'CONSIDER REDUCING SI UNITS FOR GREATER RESOLUTION.')
            step = round(step)
        self.z_step = step

    def set_polarization(self, pol):
        """
        Set the mode polarisation to be simulated ('s' or 'TE' and 'p' or 'TM')
        """
        assert pol in ['s', 'p', 'TE', 'TM'], \
            ValueError("Polarisation must one of; 's', 'p', 'TE', 'TM'")
        if self.th != 0:
            assert pol in ['s', 'p', 'TE', 'TM'], \
                ValueError("Polarisation must be defined when angle of incidence is not 0$\degree$s")
        self.pol = pol

    def set_field(self, field):
        """
        Set the field to be evaluated. Either 'E' (default) or 'H' field.
        """
        if field not in ['E', 'H']:
            raise ValueError("The field must be either 'E' of 'H'.")
        self.field = field

    def set_leaky_or_guiding(self, mode='leaky'):
        """
        Determines whether simulation will solve for a guiding mode or leaky mode.
        """
        if mode == 'leaky':
            self.guided = False
        elif mode == 'guiding':
            self.guided = True
        else:
            ValueError('options for mode must be either "leaky" or "guiding"')

    def set_incident_angle(self, th, units='radians'):
        """
        Set the incident angle of the plane wave for a leaky mode.
        """
        assert not self.guided, ValueError('Run set_leaky_or_guiding(leaky=True) first to evaluate leaky modes.')
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

    def set_guided_mode(self, n_11):
        """
        Set the normalised parallel wave vector, n_11, for the guided mode to be evaluated.
        """
        assert self.guided, \
            ValueError('Run set_leaky_or_guiding(leaky=False) first to evaluate guided modes.')
        n = self.n_list
        # See Quantum Electronics by Yariv pg.603
        assert max(n) >= n_11 > max(n[0], n[-1]), ValueError('Input n_11 is not valid for a guided mode.')
        self.n_11 = n_11

    def calc_omega(self):
        """
        Calculate the omega corresponding to lam_vac. Note units will depend on lam_vac.
        """
        return 2 * pi * c / self.lam_vac

    def calc_n_11(self):
        if not self.guided:
            # Continuous across layers, so can evaluate from input theta
            # and medium for incoming wave (hence leaky mode)
            n0 = self.n_list[0].real
            return n0 * sin(self.th)
        else:
            return self.n_11

    def calc_xi(self, j):
        """
        Normalised perpendicular wave-vector in layer j.
        """
        n = self.n_list[j]  # Normalised layer wave-vector magnitude (k(j)/k_vac)
        n_11 = self.calc_n_11()  # Normalised layer parallel wave-vector magnitude
        return sqrt(n ** 2 - n_11 ** 2)

    def calc_k(self, j):
        """
        Calculate the wave vector magnitude in layer j. Alternatively if j ==- 1
        then we calculate the vacuum wave vector (k = omega/c = 2pi/lam_vac).
        """
        if j == -1:
            n = 1
        else:
            n = self.n_list[j].real
        return n * self.k_vac

    def calc_q(self, j):
        """
        Perpendicular wave vector in layer j.
        """
        xi = self.calc_xi(j)
        return xi * self.k_vac

    def calc_k_11(self):
        """
        Parallel wave vector (same in all layers).
        """
        n_11 = self.calc_n_11()
        return n_11 * self.k_vac

    def calc_wave_vector_components(self, j):
        """
        The wave vector magnitude and it's components perpendicular and parallel
        to the interface inside the layer j.
        """
        k = self.calc_k(j)
        k_11 = self.calc_k_11()
        q = self.calc_q(j)
        return k, q, k_11

    def i_matrix(self, j, k):
        """
        Returns the interference matrix between layers j and k.
        """
        xi_j = self.calc_xi(j)
        xi_k = self.calc_xi(k)
        n_j = self.n_list[j]
        n_k = self.n_list[k]
        # Evaluate reflection and transmission coefficients for E field
        if self.pol in ['p', 'TM']:
            r = (xi_j * n_k ** 2 - xi_k * n_j ** 2) / (xi_j * n_k ** 2 + xi_k * n_j ** 2)
            t = (2 * n_j * n_k * xi_j) / (xi_j * n_k ** 2 + xi_k * n_j ** 2)
        elif self.pol in ['s', 'TE']:
            r = (xi_j - xi_k) / (xi_j + xi_k)
            t = (2 * xi_j) / (xi_j + xi_k)
        else:
            raise ValueError('A polarisation for the field must be set.')
        if self.field == 'H':
            # Convert transmission coefficient for E field to that of the H field.
            # The reflection coefficient is the same as the medium does not change.
            t *= n_k / n_j
        # TODO: need to sort this out
        if t == 0:
            t = np.nan
        assert t != 0, ValueError("Can't evaluate I_mat when transmission t==0 as 1/t == inf")
        return (1 / t) * np.array([[1, r], [r, 1]], dtype=complex)

    def l_matrix(self, j):
        """
        Returns the propagation L matrix for layer j.
        """
        qj = self.calc_q(j)
        dj = self.d_list[j]
        assert exp(-1j * qj * dj) < np.finfo(np.complex).max, ValueError('l_matrix is unstable.')
        return np.array([[exp(-1j * qj * dj), 0], [0, exp(1j * qj * dj)]], dtype=complex)

    def s_matrix(self):
        """
        Returns the total system transfer matrix s.
        """
        s = self.i_matrix(0, 1)
        for j in range(1, self.num_layers - 1):
            l = self.l_matrix(j)
            i = self.i_matrix(j, j + 1)
            s = s @ l @ i
        return s

    def s_primed_matrix(self, layer):
        """
        Returns the partial system transfer matrix s_prime.
        """
        s_prime = self.i_matrix(0, 1)
        for j in range(1, layer):
            l = self.l_matrix(j)
            i = self.i_matrix(j, j + 1)
            s_prime = s_prime @ l @ i
        return s_prime

    def s_dprimed_matrix(self, layer):
        """
        Returns the partial system transfer matrix s_dprime (doubled prime).
        """
        s_dprime = self.i_matrix(layer, layer + 1)
        for j in range(layer + 1, self.num_layers - 1):
            l = self.l_matrix(j)
            i = self.i_matrix(j, j + 1)
            s_dprime = s_dprime @ l @ i
        return s_dprime

    def layer_field_amplitudes(self, layer):
        """
        Evaluate fwd and bkwd field amplitude coefficients (E or H) in a layer.
        Coefficients are in units of the fwd incoming wave amplitude for leaky modes
        and in terms of the superstrate (j=0) outgoing wave amplitude for guided modes.
        """
        if not self.guided:
            # Calculate leaky amplitudes
            # Transfer matrix of system
            s = self.s_matrix()
            # Reflection for incoming wave incident of LHS of structure
            r = s[1, 0] / s[0, 0]
            # Evaluate lower cladding
            if layer == 0:
                field_plus = 1 + 0j
                field_minus = r
            # Evaluate upper cladding
            elif layer == self.num_layers - 1:
                field_plus = 1 / s[0, 0]
                field_minus = 0 + 0j
            # Evaluate field amplitudes in internal layers
            else:
                s_prime = self.s_primed_matrix(layer)
                # TODO: remove
                if det(s_prime) == 0:
                    return 0, 0
                assert det(s_prime) != 0, ValueError('Determinant == 0 will give inf for field coefficient.')
                field_plus = (s_prime[1, 1] - r * s_prime[0, 1]) / det(s_prime)
                field_minus = (r * s_prime[0, 0] - s_prime[1, 0]) / det(s_prime)
        else:
            # Calculate guided amplitudes
            # Evaluate lower cladding
            if layer == 0:
                field_plus = 0 + 0j
                field_minus = 1 + 0j
            # Evaluate upper cladding
            elif layer == self.num_layers - 1:
                s = self.s_matrix()
                # TODO: check why not [0, 1]
                field_plus = 1 / s[1, 0]
                field_minus = 0 + 0j
            # Evaluate field amplitudes in internal layers
            else:
                s_prime = self.s_primed_matrix(layer)
                assert det(s_prime) != 0, ValueError('Determinant == 0 will give inf for field coefficient.')
                field_plus = - s_prime[0, 1] / det(s_prime)
                field_minus = s_prime[0, 0] / det(s_prime)
        return field_plus, field_minus

    def calc_layer_field(self, layer):
        """
        Evaluate the field (E or H) as a function of z (depth) into the layer, j.
        field_plus is the forward component of the field (e.g. E_j^+)
        field_minus is the backward component of the field (e.g. E_j^-)
        """
        assert self.d_list[layer] > 0, ValueError('Layer must have a thickness to use this function.')

        # Wave vector components in layer
        k, q, k_11 = self.calc_wave_vector_components(layer)

        # z positions to evaluate field at at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        # Note field_plus and field_minus are defined at cladding-layer boundary so need to
        # propagate wave 'backwards' in the lower cladding by reversing z
        if layer == 0:
            z = -z[::-1]

        # field(z) field in terms of incident field amplitude (A_0^+)
        field_plus, field_minus = self.layer_field_amplitudes(layer)
        field = field_plus * exp(1j * q * z) + field_minus * exp(-1j * q * z)
        field_squared = abs(field) ** 2
        if self.d_list[layer] != 0:
            field_avg = sum(field_squared) / (self.z_step * self.d_list[layer])
        else:
            field_avg = np.nan
        return {'z': z, 'field': field, 'field_squared': field_squared, 'field_avg': field_avg}

    def calc_field_structure(self):
        """
        Evaluate the field at all z positions within the structure.
        """
        z = np.arange((self.z_step / 2.0), self.d_cumulative[-1], self.z_step)
        # get z_mat - specifies what layer the corresponding point in z is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z)
        comp2 = np.transpose(np.kron(np.ones((len(z), 1)), self.d_cumulative))
        z_mat = sum(comp1 > comp2, 0)

        field = np.zeros(len(z), dtype=complex)
        # Loop through all layers with a thickness (claddings with 0 thickness will not show in z_mat)
        for layer in range(min(z_mat), max(z_mat) + 1):
            # Calculate z indices inside structure for the layer
            z_indices = np.where(z_mat == layer)
            field[z_indices] = self.calc_layer_field(layer)['field']
        field_squared = abs(field) ** 2
        return {'z': z, 'field': field, 'field_squared': field_squared}

    def _s11(self, n_11):
        self.n_11 = n_11
        s = self.s_matrix()
        return s[0, 0].real

    def s11_guided(self):
        """
        Evaluate S_11=(1/t) as a function of beta (k_ll) in the guided regime.
        When S_11 = 0 the corresponding beta is a guided mode.
        """
        assert self.guided, ValueError('"self.guided" must be set to true before running this function.')
        n = self.n_list.real
        n_11_range = np.linspace(n[0], max(n), num=1000, endpoint=False)[1:]
        s_11 = np.array([])
        for n_11 in n_11_range:
            s_11 = np.append(s_11, self._s11(n_11))
        return n_11_range, s_11.real

    def calc_guided_modes(self, verbose=True, normalised=False):
        """
        Return the parallel wave vectors (k_11 or beta) of all guided modes that the structure
        supports. Array returned is arranged from lowest mode to highest mode.

        If normalised=True return (k_ll/k_vac = n_11)

        Method: Evaluates the poles of the transfer matrix (S_11=0) as a function of n_11 in the
        guided regime:  n_clad < k_ll/k < max(n), k_11/k = n_11
        """
        # Eq (11)
        # TODO: check whether comma means and/or corresponding to  max()/min()
        n = self.n_list.real
        assert np.any(n[1:-1] > max(n[0], n[-1])), ValueError('This structure does not support wave guiding.')
        assert self.guided, ValueError('"self.guided" must be set to true before running this function.')
        # Find supported guiding modes
        n_clad = max(n[0], n[-1])
        n_11 = roots(self._s11, n_clad, max(n), verbose=verbose)
        # Need to flip array to arrange from lowest to highest mode
        n_11 = n_11[::-1]

        if not normalised:
            n_11 *= self.k_vac
        return n_11

    def calc_group_velocity(self):
        lam_vac = self.lam_vac

        # Take 1% either side of the emission wavelength (using meters)
        self.set_vacuum_wavelength(int(1.01 * lam_vac))
        omega1 = self.omega
        beta_lower = self.calc_guided_modes(verbose=False, normalised=False)

        self.set_vacuum_wavelength(int(0.99 * lam_vac))
        omega2 = self.omega
        beta_upper = self.calc_guided_modes(verbose=False, normalised=False)

        assert len(beta_lower) == len(beta_upper), \
            ValueError('Number of guided modes must be equal when calculating the group velocity.')

        d_beta = beta_upper - beta_lower
        d_omega = abs(omega1 - omega2)
        vg = d_omega / d_beta

        # Reset the vacuum emission wavelength to start of function
        self.set_vacuum_wavelength(lam_vac)
        return vg

    def get_layer_boundaries(self):
        """
        Return layer boundary zs assuming that the lower cladding boundary is at z=0.
        """
        return self.d_cumulative

    def get_structure_thickness(self):
        """
        Return the structure thickness.
        """
        return self.d_cumulative[-1]

    def calc_z_to_lambda(self, z, center=True):
        """
        Convert z positions to units of wavelength (optional) from the centre.
        """
        if center:
            z -= self.get_structure_thickness() / 2
        z /= self.lam_vac
        return z

    def calc_r_and_t(self):
        """
        Return the complex reflection and transmission coefficients of the structure.
        """
        s = self.s_matrix()
        r = s[1, 0] / s[0, 0]
        t = 1 / s[0, 0]
        return r, t

    def calc_reflection_and_transmission(self, correction=True):
        """
        Return the reflectance and transmittance of the structure.
        Correction option for transmission due to beam expansion:
            https://en.wikipedia.org/wiki/Fresnel_equations
        """
        r, t = self.calc_r_and_t()
        reflection = abs(r) ** 2
        transmission = abs(t) ** 2
        if correction:
            n_1 = self.n_list[0].real
            n_2 = self.n_list[-1].real
            th_out = snell(n_1, n_2, self.th)
            rho = n_2 / n_1
            m = np.cos(th_out) / np.cos(self.th)
            transmission *= rho * m
        return reflection, transmission

    def calc_absorption(self):
        n = self.n_list
        # Absorption coefficient in 1/cm
        absorption = np.zeros(self.num_layers)
        for layer in range(1, self.num_layers):
            absorption[layer] = (4 * pi * n[layer].imag) / (self.lam_vac * 1.0e-7)
        return absorption

    def flip(self):
        """
        Flip the structure front-to-back.
        """
        self.d_list = self.d_list[::-1]
        self.n_list = self.n_list[::-1]
        self.d_cumulative = np.cumsum(self.d_list)

    def print_info(self):
        """
        Command line verbose feedback of the structure.
        """
        print('Simulation info.\n')

        print('Multi-layered Structure:')
        print('d\t\tn')
        for n, d in zip(self.n_list, self.d_list):
            print('{0:4g}\t{1:g}'.format(d, n))
        print('\nFree space wavelength: {:g}\n'.format(self.lam_vac))

    def show_structure(self):
        """
        Brings up a plot showing the structure.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from collections import OrderedDict

        # Shades to fill rectangles with based on refractive index
        alphas = abs(self.n_list) / max(abs(self.n_list))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, dx in enumerate(self.d_list[1:-1]):
            x = self.d_cumulative[i]
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
        ax.set_xlim([0, self.d_cumulative[-1]])
        ax.set(xlabel=r'x', ylabel=r'A.U.')
        plt.show()
