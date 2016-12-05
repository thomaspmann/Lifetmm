import logging
import time

import numpy as np
import scipy.integrate as integrate
from numpy import pi, sin, sum, exp, sinc, conj
from scipy.constants import c
from tqdm import *

from lifetmm.Methods.TransferMatrix import TransferMatrix

log = logging.getLogger(__name__)


class LifetimeTmm(TransferMatrix):
    def calc_spe_layer_leaky(self, layer, emission='Lower', th_pow=8):
        """
        Evaluate the spontaneous emission rates for dipoles in a layer radiating into 'Lower' or 'Upper' modes.
        Rates are normalised w.r.t. free space emission or a randomly orientated dipole.
        """
        self.set_leaky_or_guiding('leaky')

        # Option checks
        assert emission in ['Lower', 'Upper'], ValueError('Emission option must be either "Upper" or "Lower".')
        assert isinstance(th_pow, int), ValueError('th_pow must be an integer.')
        assert self.d_list[layer] > 0, ValueError('Layer must have a thickness to use this function.')

        # Flip the structure and solve using lower leaky equations for upper leaky modes.
        # Results are flipped back at the end of this function to give the correct orientation again.
        if emission == 'Upper':
            self.flip()
            layer = self.num_layers - layer - 1

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
            k, q, k_11 = self.calc_wave_vector_components(layer)

            # Check that the mode exists
            assert k_11 ** 2 <= self.calc_k(0) ** 2, ValueError('k_11 can not be larger than k0!')

            # !* TE leaky modes *!
            self.set_polarization('TE')
            # Calculate E field within layer
            self.set_field('E')
            # E field coefficients in terms of incoming amplitude
            E_plus, E_minus = self.layer_field_amplitudes(layer)
            E['TE'] = E_plus * exp(1j * q * z) + E_minus * exp(-1j * q * z)
            # Orthonormality condition: Normalise outgoing TE wave to medium refractive index.
            E['TE'] /= self.n_list[0]

            # !* TM leaky modes *!
            self.set_polarization('TM')
            # Calculate H field within layer
            self.set_field('H')
            # H field coefficients in terms of incoming amplitude
            H_plus, H_minus = self.layer_field_amplitudes(layer)
            # Calculate the electric field component perpendicular (s) to the interface
            E['TM_s'] = k_11*(H_plus * exp(1j * q * z) + H_minus * exp(-1j * q * z))
            # Calculate the electric field component parallel (p) to the interface
            E['TM_p'] = q*(H_plus * exp(1j * q * z) - H_minus * exp(-1j * q * z))

            # Take the squares of all E field components and add sin(theta) weighting
            for key in list(E.dtype.names):
                E[key] = abs(E[key]) ** 2 * sin(theta)

            # Wave vector components in upper cladding
            q_clad = self.calc_q(self.num_layers - 1)
            # Split solutions into partial and fully leaky modes
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

    def calc_spe_structure_leaky(self, th_pow=8):
        """
        Evaluate the spontaneous emission rate vs z of the structure for each dipole orientation.
        Rates are normalised w.r.t. free space emission or a randomly orientated dipole.
        """
        self.set_leaky_or_guiding('leaky')

        # z positions to evaluate E field at over entire structure
        z_pos = np.arange((self.z_step / 2.0), self.d_cumulative[-1], self.z_step)

        # get z_mat - specifies what layer the corresponding point in z_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z_pos)
        comp2 = np.transpose(np.kron(np.ones((len(z_pos), 1)), self.d_cumulative))
        z_mat = sum(comp1 > comp2, 0)

        # Structure to hold field spontaneous emission rate components over z
        spe = np.zeros(len(z_pos), dtype=[('total', 'float64'),
                                          ('parallel', 'float64'),
                                          ('perpendicular', 'float64'),
                                          ('avg', 'float64'),
                                          ('lower', 'float64'),
                                          ('upper', 'float64'),
                                          ('TE', 'float64'),
                                          ('TM_p', 'float64'),
                                          ('TM_s', 'float64'),
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

        # Calculate emission rates for leaky modes in each layer
        logging.info('Evaluating lower and upper leaky modes for each layer:')
        for layer in range(min(z_mat), max(z_mat) + 1):
            # logging.info simulation information to command line
            if layer == 0:
                logging.info('\tLayer -> lower cladding...')
            elif layer == self.num_layers - 1:
                logging.info('\tLayer -> upper cladding...')
            else:
                logging.info('\tLayer -> internal {0:d} / {1:d}...'.format(layer, self.num_layers - 2))
            time.sleep(0.2)  # Fixes progress bar occurring before text

            # Find indices corresponding to the layer we are evaluating
            ind = np.where(z_mat == layer)

            # Calculate lower leaky modes
            spe_layer = self.calc_spe_layer_leaky(layer, emission='Lower', th_pow=th_pow)['spe']
            spe['TE_lower'][ind] += spe_layer['TE']
            spe['TM_p_lower'][ind] += spe_layer['TM_p']
            spe['TM_s_lower'][ind] += spe_layer['TM_s']
            spe['TE_lower_full'][ind] += spe_layer['TE_full']
            spe['TM_s_lower_full'][ind] += spe_layer['TM_s_full']
            spe['TM_p_lower_full'][ind] += spe_layer['TM_p_full']
            spe['TE_lower_partial'][ind] += spe_layer['TE_partial']
            spe['TM_s_lower_partial'][ind] += spe_layer['TM_s_partial']
            spe['TM_p_lower_partial'][ind] += spe_layer['TM_p_partial']

            # Calculate upper leaky modes (always leaky as n[0] > n[-1])
            spe_layer = self.calc_spe_layer_leaky(layer, emission='Upper', th_pow=th_pow)['spe']
            spe['TE_upper'][ind] += spe_layer['TE']
            spe['TM_p_upper'][ind] += spe_layer['TM_p']
            spe['TM_s_upper'][ind] += spe_layer['TM_s']

        # Totals
        spe['TE'] = spe['TE_lower'] + spe['TE_upper']
        spe['TM_p'] = spe['TM_p_lower'] + spe['TM_p_upper']
        spe['TM_s'] = spe['TM_s_lower'] + spe['TM_s_upper']
        spe['lower'] = spe['TE_lower'] + spe['TM_p_lower'] + spe['TM_s_lower']
        spe['upper'] = spe['TE_upper'] + spe['TM_p_upper'] + spe['TM_s_upper']
        spe['total'] = spe['TE'] + spe['TM_p'] + spe['TM_s']
        spe['parallel'] = spe['TE'] + spe['TM_p']
        spe['perpendicular'] = spe['TM_s']

        # Average for a randomly orientated dipole
        spe['avg'] = (2 / 3) * spe['parallel'] + (1 / 3) * spe['perpendicular']

        return {'z': z_pos, 'spe': spe}

    def calc_spe_layer_guided(self, layer, roots_te=None, roots_tm=None, vg_te=None, vg_tm=None):
        assert self.d_list[layer] > 0, ValueError('Layer must have a thickness to use this function.')
        self.set_leaky_or_guiding('guiding')

        # Only re-evaluate the guided roots and v_g if not passed to function. Computationally intensive.
        # Will only be required if the function is not called from self.spe_structure_guided()
        if all(arg is None for arg in (roots_te, roots_tm, vg_te, vg_tm)):
            logging.info('Evaluating guided modes (k_11/k) and group velocity for each polarisation:')
            logging.info('Finding TE modes')
            self.set_polarization('TE')
            self.set_field('E')
            roots_te = self.calc_guided_modes(normalised=True)
            # Calculate group velocity for each mode
            logging.info('Calculating group velocity for each mode...')
            vg_te = self.calc_group_velocity()
            logging.info('Done!')
            logging.info('Finding TM modes')
            self.set_polarization('TM')
            self.set_field('H')
            roots_tm = self.calc_guided_modes(normalised=True)
            # Calculate group velocity for each mode
            logging.info('Calculating group velocity for each mode...')
            vg_tm = self.calc_group_velocity()
            logging.info('Done!')

        # z positions to evaluate E at
        z = np.arange((self.z_step / 2.0), self.d_list[layer], self.z_step)
        if layer == 0:
            # A_plus and A_minus are defined at first cladding-layer boundary.
            # Therefore must propagate waves backwards in the first cladding.
            z = -z[::-1]

        # Structure to hold field E(z) components of each mode for each dipole orientation for a given theta
        electric_field = np.zeros(len(z), dtype=[('TE', 'complex128'),
                                                 ('TM_p', 'complex128'),
                                                 ('TM_s', 'complex128')])

        # Structure to hold field SPE(z) components of each mode for each dipole orientation for a theta
        spe = np.zeros(len(z), dtype=[('total', 'float64'),
                                      ('TE', 'float64'),
                                      ('TM_p', 'float64'),
                                      ('TM_s', 'float64')])

        # !* TE guided modes *!
        self.set_polarization('TE')
        self.set_field('E')
        for n_11, v in zip(roots_te, vg_te):
            self.set_guided_mode(n_11)
            # Evaluate the normalisation (B4) and apply
            norm = 0
            for j in range(0, self.num_layers):
                k, q, k_11 = self.calc_wave_vector_components(j)
                a, b = self.layer_field_amplitudes(j)
                if j == 0:
                    chi = np.imag(q)
                    norm += abs(b) ** 2 * (chi ** 2 + k_11 ** 2) / (2 * chi)
                elif j == (self.num_layers - 1):
                    chi = np.imag(q)
                    norm += abs(a) ** 2 * (chi ** 2 + k_11 ** 2) / (2 * chi)
                else:
                    dj = self.d_list[j]
                    w1 = (k_11 ** 2 + q * conj(q)) * sinc((q - conj(q)) * dj / 2)
                    w2 = (k_11 ** 2 - q * conj(q)) * sinc((q + conj(q)) * dj / 2)
                    norm += dj * (w1 * (abs(a) ** 2 + abs(b) ** 2) + w2 * (conj(a) * b + conj(b) * a))
            assert np.isreal(norm) and norm > 0, ValueError('Check Normalisation - should be real and > 0')
            norm = 1 / np.sqrt(np.real(norm))

            # E field coefficients in terms of layer 0 (superstrate) outgoing field amplitude
            a, b = self.layer_field_amplitudes(layer)
            a *= norm
            b *= norm

            # Wave vector components in layer
            k, q, k_11 = self.calc_wave_vector_components(layer)
            assert k_11 == (self.n_11 * self.k_vac), ValueError('Check TE')

            # Evaluate E(z)
            electric_field['TE'] = a * exp(1j * q * z) + b * exp(-1j * q * z)
            spe['TE'] += abs(electric_field['TE']) ** 2 * (k_11 / v)
        # Normalise emission rates to vacuum emission rate of a randomly orientated dipole
        spe['TE'] *= 3 * pi * c / 4

        # !* TM guided modes *!
        self.set_polarization('TM')
        self.set_field('H')
        # Find guided modes parallel wave vector
        for n_11, v in zip(roots_tm, vg_tm):
            self.set_guided_mode(n_11)

            # Evaluate the normalisation (B8) and apply
            norm = 0
            for j in range(0, self.num_layers):
                k, q, k_11 = self.calc_wave_vector_components(j)
                a, b = self.layer_field_amplitudes(j)
                if j == 0:
                    chi = np.imag(q)
                    norm += abs(b) ** 2 / (2 * chi)
                elif j == (self.num_layers - 1):
                    chi = np.imag(q)
                    norm += abs(a) ** 2 / (2 * chi)
                else:
                    dj = self.d_list[j]
                    w1 = sinc((q - conj(q)) * dj / 2)
                    w2 = sinc((q + conj(q)) * dj / 2)
                    norm += dj * (w1 * (abs(a) ** 2 + abs(b) ** 2) + w2 * (conj(a) * b + conj(b) * a))
            assert np.isreal(norm), ValueError('Check Normalisation - should be real')

            # E field coefficients in terms of layer 0 (superstrate) outgoing field amplitude
            a, b = self.layer_field_amplitudes(layer)
            a /= np.sqrt(norm)
            b /= np.sqrt(norm)

            # Wave vector components in layer (q, k_11 are angle dependent)
            k, q, k_11 = self.calc_wave_vector_components(layer)
            assert k_11 == (self.n_11 * self.k_vac), ValueError('Check TM')

            # Calculate the electric field component perpendicular (s) and parallel (p) to the interface
            eps_j = self.n_list[layer].real ** 2
            electric_field['TM_s'] = (1j * k_11 / eps_j) * (a * exp(1j * q * z) + b * exp(-1j * q * z))
            electric_field['TM_p'] = (1j * q / eps_j) * (-a * exp(1j * q * z) + b * exp(-1j * q * z))

            spe['TM_s'] += abs(electric_field['TM_s']) ** 2 * (k_11 / v)
            spe['TM_p'] += abs(electric_field['TM_p']) ** 2 * (k_11 / v)

        # Normalise emission rates to vacuum emission rate of a randomly orientated dipole
        spe['TM_s'] *= (3 * c * self.lam_vac ** 4) / (2 ** 5 * pi ** 3)
        spe['TM_p'] *= (3 * c * self.lam_vac ** 4) / (2 ** 6 * pi ** 3)

        return {'z': z, 'spe': spe}

    def calc_spe_structure_guided(self):
        self.set_leaky_or_guiding('guiding')

        # z positions to evaluate E field at over entire structure
        z_pos = np.arange((self.z_step / 2.0), self.d_cumulative[-1], self.z_step)

        # get z_mat - specifies what layer the corresponding point in z_pos is in
        comp1 = np.kron(np.ones((self.num_layers, 1)), z_pos)
        comp2 = np.transpose(np.kron(np.ones((len(z_pos), 1)), self.d_cumulative))
        z_mat = sum(comp1 > comp2, 0)

        # Structure to hold field spontaneous emission rate components over z
        spe = np.zeros(len(z_pos), dtype=[('TE', 'float64'),
                                          ('TM_p', 'float64'),
                                          ('TM_s', 'float64'),
                                          ('parallel', 'float64'),
                                          ('perpendicular', 'float64'),
                                          ('avg', 'float64')])

        logging.info('Evaluating guided modes (k_11/k) and group velocity for each polarisation:')
        logging.info('Finding TE modes')
        self.set_polarization('TE')
        self.set_field('E')
        roots_te = self.calc_guided_modes(normalised=True)
        # Calculate group velocity for each mode
        logging.info('Calculating group velocity for each mode...')
        vg_te = self.calc_group_velocity()
        logging.info('Done!')
        logging.info('Finding TM modes')
        self.set_polarization('TM')
        self.set_field('H')
        roots_tm = self.calc_guided_modes(normalised=True)
        # Calculate group velocity for each mode
        logging.info('Calculating group velocity for each mode...')
        vg_tm = self.calc_group_velocity()
        logging.info('Done!')

        logging.info('Evaluating guided mode spontaneous emission profiles:')
        for layer in range(min(z_mat), max(z_mat) + 1):
            # logging.info simulation information to command line
            if layer == 0:
                logging.info('\tLayer -> lower cladding...')
            elif layer == self.num_layers - 1:
                logging.info('\tLayer -> upper cladding...')
            else:
                logging.info('\tLayer -> internal {0:d} / {1:d}...'.format(layer, self.num_layers - 2))
            time.sleep(0.2)  # Fixes progress bar occurring before text

            # Find indices corresponding to the layer we are evaluating
            ind = np.where(z_mat == layer)

            # Calculate lower leaky modes
            spe_layer = self.calc_spe_layer_guided(layer, roots_te, roots_tm, vg_te, vg_tm)['spe']
            spe['TE'][ind] += spe_layer['TE']
            spe['TM_p'][ind] += spe_layer['TM_p']
            spe['TM_s'][ind] += spe_layer['TM_s']

        # Totals
        spe['parallel'] = spe['TE'] + spe['TM_p']
        spe['perpendicular'] = spe['TM_s']

        # Average for a randomly orientated dipole
        spe['avg'] = (2 / 3) * spe['parallel'] + (1 / 3) * spe['perpendicular']

        return {'z': z_pos, 'spe': spe}

    def calc_spe_structure(self, th_pow=10):
        logging.info("Calculating leaky modes...")
        result = self.calc_spe_structure_leaky(th_pow=th_pow)
        leaky = result['spe']
        z = result['z']

        n = self.n_list.real
        if np.any(n[1:-1] > max(n[0], n[-1])):
            logging.info("Structure suppports waveguiding. Calculating guided modes...")
            guided = self.calc_spe_structure_guided()['spe']
            return {'z': z, 'leaky': leaky, 'guided': guided}
        else:
            logging.info("Structure does not support waveguiding.")
            return {'z': z, 'leaky': leaky}


# Helper Functions
def flip_spe_results(spe):
    """
    Flip the array containing the spe result
    """
    for key in spe.dtype.names:
        spe[key] = spe[key][::-1]
    return spe
