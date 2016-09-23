import numpy as np
import scipy.integrate as integrate

# from tqdm import *
from numpy import pi, sin, sum
from lifetmm.Methods.TransferMatrix import TransferMatrix


class LifetimeTmm(TransferMatrix):
    # def __init__(self):
    #     super().__init__()
    #     self.time_rev = False

    def spe_layer(self, layer):
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

        # for i, theta in tqdm(enumerate(theta_input), **kwargs):
        for i, theta in enumerate(theta_input):
            self.set_angle(theta)

            # Calculate E field within layer
            E = self.layer_E_field(layer)['E']

            # Normalise for outgoing wave medium refractive index - only TE
            if self.pol in ['s', 'TE']:
                if self.radiative == 'Lower':
                    E /= self.n_list[0].real
                else:  # radiative == 'Upper'
                    E /= self.n_list[-1].real

            # TODO: TM Mode check
            # # Wave vector components in layer
            # k, k_z, k_11 = self.wave_vector(layer)
            # if self.pol in ['p', 'TE']:
            #     if self.dipole == 'Vertical':
            #         E *= k_11
            #     else:  # self.dipole == 'Horizontal'
            #         E *= k_z

            E_square_theta[i, :] += abs(E)**2 * sin(theta)

        # Evaluate spontaneous emission rate
        # (axis=0 integrates all rows, containing thetas, over each columns, z)
        spe = integrate.romb(E_square_theta, dx=dth, axis=0)

        # Outgoing E mode refractive index weighting (3)
        if self.radiative == 'Lower':
            spe *= self.n_list[0].real ** 3
        else:  # radiative == 'Upper'
            spe *= self.n_list[-1].real ** 3

        # Normalise to vacuum emission rate of a randomly orientated dipole
        spe *= 3/8
        # TODO: TM Mode check
        # if self.pol in ['p', 'TE']:
        #     spe *= ((self.lam_vac*1E-9)**2) / (4 * pi**2 * self.n_list[layer].real ** 4)
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
            # self.radiative = 'Upper'
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
