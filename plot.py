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
    r0 = h5py.File(filename, 'r')['run_config']['domain_radius'][()]
    return [-r0, r0, -r0, r0]




def conserved(filename, field):
    """
    Reconstruct a uniform array for one of the conserved fields, from the blocks
    in a checkpoint file.
    """
    blocks = []
    h5f = h5py.File(filename, 'r')
    print('loading {}/{}'.format(filename, field))
    for block in h5f['conserved']:
        level, rest = block.split(':')
        index = [int(i) for i in rest.split('-')]
        blocks.append((index, h5f['conserved'][block]))
    nb = h5f['run_config']['num_blocks'][()]
    bs = h5f['run_config']['block_size'][()]
    result = np.zeros([bs * nb, bs * nb])
    for (i, j), v in blocks:
        result[i*bs:i*bs+bs, j*bs:j*bs+bs] = v[...][:,:,field]
    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    rho = conserved(args.filename, 0)
    plt.imshow(np.log10(rho), cmap='inferno', extent=extent(args.filename))
    plt.colorbar()
    plt.show()
