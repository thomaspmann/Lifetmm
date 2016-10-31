import time

import numpy as np
import scipy.integrate as integrate
from numpy import pi, sin, sum, exp, sinc
from scipy.constants import c
from tqdm import *

from lifetmm.Methods.TransferMatrix import TransferMatrix


class LifetimeTmm(TransferMatrix):

    def spe_layer_radiative(self, layer, emission='Lower', th_pow=8):
        """ Evaluate the spontaneous emission rates for dipoles in a layer radiating into 'Lower' or 'Upper' modes.
        Rates are normalised w.r.t. free space emission or a randomly orientated dipole.
        """
        # Option checks
        assert emission in ['Lower', 'Upper'], \
            ValueError('Emission option must be either "Upper" or "Lower".')
        assert isinstance(th_pow, int), ValueError('pow must be an integer.')

        # Flip the structure and solve using lower radiative equations for upper radiative modes.
        # Results are flipped back at the end of this function to give the correct orientation again.
        if emission == 'Upper':
            self.flip()
            layer = self.num_layers - layer - 1

        # Lower cladding incoming wave vector magnitude
        k0 = self.k(0)

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        if layer == 0:
            # A_plus and A_minus are defined at first cladding-layer boundary.
            # Therefore must propagate waves backwards in the first cladding.
            z = -z[::-1]

        # Angles of emission to simulate over.
        # Note: don't include pi/2 as then transmission and reflection do not make sense.
        # res for linspace must have this form for the simpsons integration later. Can change the power.
        res = 2 ** th_pow + 1
        th_in, dth = np.linspace(0, pi / 2, num=res, endpoint=False, retstep=True)

        # Structure to hold field E(z) components of each mode for each dipole orientation for a given theta
        E = np.zeros(len(z), dtype=[('TE', 'complex128'),
                                    ('TM_p', 'complex128'),
                                    ('TM_s', 'complex128')])

        # Arrays to store the square of the E fields for all thetas and z
        E2_z_th = np.zeros((len(th_in), len(z)), dtype=[('TE_full', 'float64'),
                                                        ('TM_p_full', 'float64'),
                                                        ('TM_s_full', 'float64'),
                                                        ('TE_partial', 'float64'),
                                                        ('TM_p_partial', 'float64'),
                                                        ('TM_s_partial', 'float64')])

        # Structure to hold field SPE(z) components of each mode for each dipole orientation for a theta
        spe = np.zeros(len(z), dtype=[('total', 'float64'),
                                      ('TE', 'float64'),
                                      ('TM_p', 'float64'),
                                      ('TM_s', 'float64'),
                                      ('TE_full', 'float64'),
                                      ('TM_p_full', 'float64'),
                                      ('TM_s_full', 'float64'),
                                      ('TE_partial', 'float64'),
                                      ('TM_p_partial', 'float64'),
                                      ('TM_s_partial', 'float64')])

        # Params for tqdm progress bar
        kwargs = {'total': res, 'unit': ' theta', 'unit_scale': True}
        # Evaluate all E field components for TE and TM modes looping over the emission angles.
        for i, theta in tqdm(enumerate(th_in), **kwargs):
            # Set the angle to be evaluated
            self.set_incident_angle(theta)

            # Wave vector components in layer (q, k_11 are angle dependent)
            k, q, k_11 = self.wave_vector(layer)

            # Check that the mode exists
            assert k_11**2 < k0**2, ValueError('k_11 can not be larger than k0!')

            # !* TE radiative modes *!
            self.set_polarization('TE')
            # Calculate E field within layer
            self.set_field('E')
            # E field coefficients in terms of incoming amplitude
            E_plus, E_minus = self.amplitude_coefficients(layer)
            E['TE'] = E_plus * exp(1j * q * z) + E_minus * exp(-1j * q * z)
            # Orthonormality condition: Normalise outgoing TE wave to medium refractive index.
            E['TE'] /= self.n_list[0]

            # !* TM radiative modes *!
            self.set_polarization('TM')
            # Calculate H field within layer
            self.set_field('H')
            # H field coefficients in terms of incoming amplitude
            H_plus, H_minus = self.amplitude_coefficients(layer)
            # Calculate the electric field component perpendicular (s) to the interface
            E['TM_s'] = k_11*(H_plus * exp(1j * q * z) + H_minus * exp(-1j * q * z))
            # Calculate the electric field component parallel (p) to the interface
            E['TM_p'] = q*(H_plus * exp(1j * q * z) - H_minus * exp(-1j * q * z))

            # Check that results seem reasonable - TMM can be unstable for large z with exponentially growing waves
            assert max(E['TE']) < 100, ValueError('TMM Unstable.')
            assert max(E['TM_s']) < 100, ValueError('TMM Unstable.')
            assert max(E['TM_p']) < 100, ValueError('TMM Unstable.')

            # Take the squares of all E field components and add sin(theta) weighting
            for key in list(E.dtype.names):
                E[key] = abs(E[key]) ** 2 * sin(theta)

            # Wave vector components in upper cladding
            q_clad = self.q(self.num_layers-1)
            # Split solutions into partial and fully radiative modes
            if np.iscomplex(q_clad):
                E2_z_th['TE_partial'][i, :] += E['TE'].real
                E2_z_th['TM_p_partial'][i, :] += E['TM_p'].real
                E2_z_th['TM_s_partial'][i, :] += E['TM_s'].real
            else:
                E2_z_th['TE_full'][i, :] += E['TE'].real
                E2_z_th['TM_p_full'][i, :] += E['TM_p'].real
                E2_z_th['TM_s_full'][i, :] += E['TM_s'].real

        for key in list(E2_z_th.dtype.names):
            # Evaluate spontaneous emission rate for each z (columns) over all thetas (rows)
            spe[key] = integrate.romb(E2_z_th[key], dx=dth, axis=0)
            # Outgoing mode refractive index weighting (between summation over j=0,M+1 and integral -> eps_j** 3/2)
            spe[key] *= self.n_list[0].real ** 3

        # Normalise emission rates to vacuum emission rate of a randomly orientated dipole
        nj = self.n_list[layer].real
        for key in list(E2_z_th.dtype.names):
            if 'TE' in key:
                spe[key] *= 3/8
            elif 'TM_p' in key:
                spe[key] *= 3 / (8 * (nj * k) ** 2)
            elif 'TM_s' in key:
                spe[key] *= 3 / (4 * (nj * k) ** 2)

        # Total emission rates
        spe['TE'] = spe['TE_full'] + spe['TE_partial']
        spe['TM_p'] = spe['TM_p_full'] + spe['TM_p_partial']
        spe['TM_s'] = spe['TM_s_full'] + spe['TM_s_partial']
        spe['total'] = spe['TE'] + spe['TM_p'] + spe['TM_s']
        # Flip structure and results back to original orientation
        if emission == 'Upper':
            self.flip()
            for key in list(spe.dtype.names):
                spe[key] = spe[key][::-1]

        return {'z': z, 'spe': spe}

    def calc_spe_structure_radiative(self):
        """ Evaluate the spontaneous emission rate vs z of the structure for each dipole orientation.
            Rates are normalised w.r.t. free space emission or a randomly orientated dipole.
        """
        # z positions to evaluate E field at over entire structure
        z_pos = np.arange((self.z_step / 2.0), self.d_cumulative[-1], self.z_step)

        # get z_mat - specifies what layer the corresponding point in z_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z_pos)
        comp2 = np.transpose(np.kron(np.ones((len(z_pos), 1)), self.d_cumulative))
        z_mat = sum(comp1 > comp2, 0)

        # Structure to hold field spontaneous emission rate components over z
        spe = np.zeros(len(z_pos), dtype=[('total', 'float64'),
                                          ('total_lower', 'float64'),
                                          ('total_upper', 'float64'),
                                          ('TE_total', 'float64'),
                                          ('TM_p_total', 'float64'),
                                          ('TM_s_total', 'float64'),
                                          ('TE_lower', 'float64'),
                                          ('TM_p_lower', 'float64'),
                                          ('TM_s_lower', 'float64'),
                                          ('TE_lower_full', 'float64'),
                                          ('TM_p_lower_full', 'float64'),
                                          ('TM_s_lower_full', 'float64'),
                                          ('TE_lower_partial', 'float64'),
                                          ('TM_p_lower_partial', 'float64'),
                                          ('TM_s_lower_partial', 'float64'),
                                          ('TE_upper', 'float64'),
                                          ('TM_p_upper', 'float64'),
                                          ('TM_s_upper', 'float64')])

        # Calculate emission rates for radiative modes in each layer
        print('Evaluating lower and upper radiative modes for each layer:')
        for layer in range(self.num_layers):
            # Print simulation information to command line
            if layer == 0:
                print('\tLayer -> lower cladding...')
            elif layer == self.num_layers - 1:
                print('\tLayer -> upper cladding...')
            else:
                print('\tLayer -> internal {0:d} / {1:d}...'.format(layer, self.num_layers-2))
            time.sleep(0.2)  # Fixes progress bar occurring before text

            # Find indices corresponding to the layer we are evaluating
            ind = np.where(z_mat == layer)

            # Calculate lower radiative modes
            spe_layer = self.spe_layer_radiative(layer, emission='Lower')['spe']
            spe['TE_lower'][ind] += spe_layer['TE']
            spe['TM_p_lower'][ind] += spe_layer['TM_p']
            spe['TM_s_lower'][ind] += spe_layer['TM_s']
            spe['TE_lower_full'][ind] += spe_layer['TE_full']
            spe['TM_s_lower_full'][ind] += spe_layer['TM_s_full']
            spe['TM_p_lower_full'][ind] += spe_layer['TM_p_full']
            spe['TE_lower_partial'][ind] += spe_layer['TE_partial']
            spe['TM_s_lower_partial'][ind] += spe_layer['TM_s_partial']
            spe['TM_p_lower_partial'][ind] += spe_layer['TM_p_partial']

            # Calculate upper radiative modes (always radiative as n[0] > n[-1])
            spe_layer = self.spe_layer_radiative(layer, emission='Upper')['spe']
            spe['TE_upper'][ind] += spe_layer['TE']
            spe['TM_s_upper'][ind] += spe_layer['TM_s']
            spe['TM_p_upper'][ind] += spe_layer['TM_p']

        # Totals
        spe['TE_total'] = spe['TE_lower'] + spe['TE_upper']
        spe['TM_s_total'] = spe['TM_s_lower'] + spe['TM_s_upper']
        spe['TM_p_total'] = spe['TM_p_lower'] + spe['TM_p_upper']
        spe['total_lower'] = spe['TE_lower'] + spe['TM_p_lower'] + spe['TM_s_lower']
        spe['total_upper'] = spe['TE_upper'] + spe['TM_p_upper'] + spe['TM_s_upper']
        # TODO: check exactly why i should be dividing by to (averaging)
        spe['total'] = (spe['total_lower'] + spe['total_upper']) / 2

        return {'z': z_pos, 'spe': spe}

    def spe_layer_guided(self, layer):
        self.guided = True
        # Evaluate guiding layer in structure(one with highest refractive index)
        n = self.n_list
        layer_guiding = np.where(n == max(n))[0][0]

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        if layer == 0:
            # A_plus and A_minus are defined at first cladding-layer boundary.
            # Therefore must propagate waves backwards in the first cladding.
            z = -z[::-1]

        # Structure to hold field E(z) components of each mode for each dipole orientation for a given theta
        E = np.zeros(len(z), dtype=[('TE', 'complex128'),
                                    ('TM_p', 'complex128'),
                                    ('TM_s', 'complex128')])

        # Structure to hold field SPE(z) components of each mode for each dipole orientation for a theta
        spe = np.zeros(len(z), dtype=[('total', 'float64'),
                                      ('TE', 'float64'),
                                      ('TM_p', 'float64'),
                                      ('TM_s', 'float64')])

        # !* TE guided modes *!
        self.set_polarization('TE')
        # Calculate E field within layer
        self.set_field('E')
        # Find guided modes parallel wave vector
        for alpha in self.find_guided_modes():
            self.n_11 = alpha
            # Wave vector components in layer (q, k_11 are angle dependent)
            k, q, k_11 = self.wave_vector(layer)
            assert k_11 == (self.n_11*self.k(-1)), ValueError('Check TE')

            # TODO: Find corresponding phase velocity
            v = 1

            # E field coefficients in terms of incoming amplitude
            E_plus, E_minus = self.amplitude_coefficients(layer)
            # Correct amplitudes with the orthonormality condition (B4)
            chi_0 = self.q(0).imag
            B_0 = self.amplitude_coefficients(0)
            norm = (chi_0**2+k_11**2)/(2*chi_0)

            chi_m_plus_1 = self.q(self.num_layers - 1).real

            # Evaluate E(z)
            E['TE'] = E_plus * exp(1j * q * z) + E_minus * exp(-1j * q * z)
            assert max(E['TE']) < 100, ValueError('TMM likely unstable.')
            E['TE'] = abs(E['TE']) ** 2 * (k / v)
            spe['TE'] += E['TE']
        # Normalise emission rates to vacuum emission rate of a randomly orientated dipole
        spe['TE'] *= 3*pi*c/4

        # !* TM radiative modes *!
        self.set_polarization('TM')
        # Calculate H field within layer
        self.set_field('H')
        # Find guided modes parallel wave vector
        for alpha in self.find_guided_modes():
            self.n_11 = alpha
            # Wave vector components in layer (q, k_11 are angle dependent)
            k, q, k_11 = self.wave_vector(layer)
            assert k_11 == (self.n_11*self.k(-1)), ValueError('Check TM')

            # TODO: Find corresponding phase velocity
            v = 1

            # H field coefficients in terms of incoming amplitude
            H_plus, H_minus = self.amplitude_coefficients(layer)
            # Calculate the electric field component perpendicular (s) to the interface
            E['TM_s'] = k_11 * (H_plus * exp(1j * q * z) + H_minus * exp(-1j * q * z))
            # Calculate the electric field component parallel (p) to the interface
            E['TM_p'] = q * (H_plus * exp(1j * q * z) - H_minus * exp(-1j * q * z))
            assert max(E['TM_s']) < 100, ValueError('TMM Unstable.')
            assert max(E['TM_p']) < 100, ValueError('TMM Unstable.')
            E['TM_s'] = abs(E['TM_s']) ** 2 * (k / v)
            E['TM_p'] = abs(E['TM_p']) ** 2 * (k / v)
            spe['TM_s'] += E['TM_s']
            spe['TM_p'] += E['TM_p']
        # Normalise emission rates to vacuum emission rate of a randomly orientated dipole
        # TODO: omega
        omega = 1
        spe['TM_s'] *= (3*pi*c**5)/(4*omega**4)
        spe['TM_p'] *= (3*pi*c**5)/(2*omega**4)

        return {'z': z, 'E': E}

    def spe_structure_guided(self):
        # Calculate emission rates for guided modes in each layer
        print('Evaluating lower and upper radiative modes for each layer:')
        for layer in range(self.num_layers):
            # Print simulation information to command line
            if layer == 0:
                print('\tLayer -> lower cladding...')
            elif layer == self.num_layers - 1:
                print('\tLayer -> upper cladding...')
            else:
                print('\tLayer -> internal {0:d} / {1:d}...'.format(layer, self.num_layers-2))
            time.sleep(0.2)  # Fixes progress bar occurring before text


# Helper Functions
def flip_spe_results(spe):
    """ Flip the spe result
    """
    for key in spe.dtype.names:
        spe[key] = spe[key][::-1]
    return spe
