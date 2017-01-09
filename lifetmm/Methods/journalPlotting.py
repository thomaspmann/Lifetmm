import matplotlib.pyplot as plt
import numpy as np


def update():
    # Journal plotting parameters
    plt.style.use('https://raw.githubusercontent.com/mn14tm/Notebooks/master/journalThomas.mplstyle')

    # http://blog.dmcdougall.co.uk/publication-ready-the-first-time-beautiful-reproducible-plots-with-matplotlib/
    WIDTH = 246.0  # the number latex spits out when typing: \the\linewidth
    FACTOR = 0.9  # the fraction of the width you'd like the figure to occupy
    fig_width_pt = WIDTH * FACTOR

    inches_per_pt = 1.0 / 72.27
    golden_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

    fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio  # figure height in inches
    fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list

    # Update rcParams for figure size
    params = {
        'figure.figsize': fig_dims,
    }
    plt.rcParams.update(params)
