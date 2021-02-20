#!/usr/bin/env python3

import argparse
import numpy as np
import cdc_loader
import matplotlib.pyplot as plt


fields = ['sigma', 'velocity_x', 'velocity_y', 'pressure', 'specific_internal_energy', 'mach_number']


def pcolormesh_data(app, field, transform=lambda x: x):
    result = []
    for key, data in getattr(app.state, field).items():
        x, y = app.mesh[key].vertices
        result.append((x, y, transform(data.T)))
    return result


def plot_field(ax, filename, field, vmin=None, vmax=None, transform=lambda x: x):
    app = cdc_loader.app(filename)
    pcm_data = pcolormesh_data(app, field, transform=transform)

    vmin = min(p[2].min() for p in pcm_data) if vmin is None else vmin
    vmax = max(p[2].max() for p in pcm_data) if vmax is None else vmax

    for (x, y, z) in pcm_data:
        cm = ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax)

    ax.set_aspect('equal')
    return cm


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
