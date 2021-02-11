#!/usr/bin/env python3

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt




def extent(filename):
    """
    Return the domain extent for a checkpoint named `filename` in the format
    expected by imshow
    """
    r0 = h5py.File(filename, 'r')['model']['domain_radius'][()]
    return [-r0, r0, -r0, r0]




def conserved(filename, field):
    """
    Reconstruct a uniform array for one of the conserved fields, from the blocks
    in a checkpoint file.
    """
    blocks = []
    h5f = h5py.File(filename, 'r')
    print('loading {}/{}'.format(filename, field))
    for block in h5f['state']['solution']:
        level, rest = block.split(':')
        index = [int(i) for i in rest.split('-')]
        blocks.append((index, h5f['state']['solution'][block]['conserved']))
    nb = h5f['model']['num_blocks'][()]
    bs = h5f['model']['block_size'][()]
    result = np.zeros([bs * nb, bs * nb])
    for (i, j), v in blocks:
        result[i*bs:i*bs+bs, j*bs:j*bs+bs] = v[()][str(field)]
    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument('--range', default='None,None', help='vmin and vmax parameters for the relief plot')
    args = parser.parse_args()

    vmin, vmax = eval(args.range)

    for filename in args.filenames:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        rho = conserved(filename, 0)
        ax1.imshow(np.log10(rho).T, cmap='inferno', origin='lower', extent=extent(filename), vmin=vmin, vmax=vmax)
        # plt.savefig(filename + '.jpg', dpi=600, format='jpg')
        # plt.colorbar()
    plt.show()
