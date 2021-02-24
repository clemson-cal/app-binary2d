#!/usr/bin/env python3

import argparse
import numpy as np
import cdc_loader
import matplotlib.pyplot as plt


fields = ['sigma', 'velocity_x', 'velocity_y', 'pressure', 'specific_internal_energy', 'mach_number']


def reconstitute(app, field):
    nb = app.config['mesh']['num_blocks']
    bs = app.config['mesh']['block_size']
    result = np.zeros([bs * nb, bs * nb])
    for (i, j), data in getattr(app.state, field).items():
        result[i*bs:i*bs+bs, j*bs:j*bs+bs] = data
    return result


def plot_field(ax, filename, field, vmin=None, vmax=None, transform=lambda x: x):
    app = cdc_loader.app(filename)
    rd = app.config['mesh']['domain_radius']
    return ax.imshow(transform(reconstitute(app, field)).T, vmin=vmin, vmax=vmax, extent=[-rd, rd, -rd, rd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument('--field', '-f', default='sigma', choices=fields, help='the hydrodynamic field to plot')
    parser.add_argument('--range', default='None,None', help='vmin and vmax parameters for the relief plot')
    parser.add_argument('--log', '-l', action='store_true', help='use log10 scaling')
    args = parser.parse_args()

    vmin, vmax = eval(args.range)

    for filename in args.filenames:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        cm = plot_field(ax1, filename, args.field, vmin=vmin, vmax=vmax, transform=np.log10 if args.log else lambda x: x)
        fig.colorbar(cm)
    plt.show()
