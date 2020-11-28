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
    for block in h5f['state']['conserved']:
        level, rest = block.split(':')
        index = [int(i) for i in rest.split('-')]
        blocks.append((index, h5f['state']['conserved'][block]))
    nb = h5f['model']['num_blocks'][()]
    bs = h5f['model']['block_size'][()]
    result = np.zeros([bs * nb, bs * nb])
    for (i, j), v in blocks:
        result[i*bs:i*bs+bs, j*bs:j*bs+bs] = v[...][:,:,field]
    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    rho = conserved(args.filename, 0)
    plt.imshow(np.log10(rho).T, cmap='inferno', origin='lower', extent=extent(args.filename), vmin=-3, vmax=1.5)
    plt.colorbar()
    plt.show()
