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


def plot_radial_profile(ax, filename, field):
    app = cdc_loader.app(filename)
    data = getattr(app.state, field)

    rs = []
    zs = []
    rd = app.config['mesh']['domain_radius']

    for i, j in app.mesh.indexes:
        x, y = app.mesh[(i, j)].vertices
        xc = 0.5 * (x[1:] + x[:-1])
        yc = 0.5 * (y[1:] + y[:-1])
        x, y = np.meshgrid(xc, yc)
        rs += np.sqrt(x**2 + y**2).tolist()
        zs += data[(i, j)].tolist()

    bins = np.linspace(0.0, rd, 128)
    z_binned, _ = np.histogram(rs, bins=bins, weights=zs)
    n_binned, _ = np.histogram(rs, bins=bins)
    n_binned[n_binned == 0] = 1
    profile = z_binned / n_binned
    ax.plot(bins[:-1], profile, label=filename)


def plot_relief(ax, filename, field, vmin=None, vmax=None, transform=lambda x: x):
    app = cdc_loader.app(filename)
    rd = app.config['mesh']['domain_radius']
    return ax.imshow(transform(reconstitute(app, field)).T, vmin=vmin, vmax=vmax, extent=[-rd, rd, -rd, rd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument('--field', '-f', default='sigma', choices=fields, help='the hydrodynamic field to plot')
    parser.add_argument('--range', default='None,None', help='vmin and vmax parameters for the relief plot')
    parser.add_argument('--log', '-l', action='store_true', help='use log10 scaling')
    parser.add_argument('--plot', default='relief', choices=['relief', 'profile'])
    args = parser.parse_args()

    vmin, vmax = eval(args.range)

    if args.plot == 'relief':
        for filename in args.filenames:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            cm = plot_relief(ax1, filename, args.field, vmin=vmin, vmax=vmax, transform=np.log10 if args.log else lambda x: x)
            fig.colorbar(cm)

    if args.plot == 'profile':
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        for filename in args.filenames:
            plot_radial_profile(ax1, filename, args.field)
        ax1.legend()

    plt.show()
