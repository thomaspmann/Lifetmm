import os
from os.path import join, isfile
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors


def loadSample(fname, matHeader=22):
    """
    Load the spectrum of the samples. Data file in form of
    :param matName: name of the sample to load data from
    :param matDir: Directory name containing the data files to be loaded
    :return:
    """
    # baseDir = '/Users/Thomas/Dropbox/PhD/Programming/Lifetmm/lifetmm'
    # baseDir ='M:\PhD\SimulationCodes\Lifetmm\lifetmm'

    fname = join(baseDir, matDir, fname)
    spectrumData = np.genfromtxt(fname, delimiter=',', dtype=float, skip_header=matHeader)
    wavelength = spectrumData[:, 0]
    intensity = spectrumData[:, 1]

    return wavelength, intensity

baseDir = '/Users/Thomas/Dropbox/PhD/Programming/Lifetmm/lifetmm'
matDir = 'samples/Temperature/Spectrum'

samples = []
files = os.listdir(join(baseDir, matDir))
sortedFiles = sorted(files, key=lambda x: int(x[0:-5]))
# sortedFiles = sorted(files, key=lambda x: int(x[1:-4]))

NUM_COLORS = len(files)
cm = plt.get_cmap('rainbow')
cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

splot = ['293K.txt', '300K.txt', '310K.txt']

for file in sortedFiles:
    if fnmatch.fnmatch(file, '*.txt'):
        wavelength, intensity = loadSample(file)
        ax.plot(wavelength, intensity, label=file[:-4])

plt.legend(prop={'size': 12}, ncol=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (A.U.')
plt.title('Spectrum of T-series samples')
# fig.savefig('moreColors.png')
plt.show()
