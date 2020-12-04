#!/usr/bin/env python3

"""
This script takes in a checkpoint file and up-samples each cell in each block
into four cells. The grid spacing is reduced by a factor of two compared to the
older file.

Authors: Zhongtian Hu    Clemson University (2020)
         Jonathan Zrake  Clemson University (2020)
"""

import argparse
import numpy as np
import h5py

default_outfile = "chkpt.upsampled.h5"
parser = argparse.ArgumentParser()
parser.add_argument("input", help='Input name of the HDF5 checkpoint file')
parser.add_argument("--output", default=default_outfile, metavar=default_outfile, help="Output file name")
parser.add_argument("--clobber", action='store_true', help='Overwrite the output file if it exists')
args = parser.parse_args()

old_file = h5py.File(args.input, 'r')
old_bs = old_file['model']['block_size'][()]
new_bs = old_bs * 2
num_blocks = old_file['model']['num_blocks'][()]

print("Checkpoint file provided is ... {}".format(args.input))
print("Output file will be named ..... {}".format(args.output))
print("Number of blocks is ........... {}".format(num_blocks))
print("Old block size is ............. {}".format(old_bs))
print("New block size will be ........ {}".format(new_bs))

try:
    new_file = h5py.File(args.output, 'w' if args.clobber else 'w-')
except OSError as e:
    print(e)
    exit()

for group in old_file:
    new_file.copy(old_file[group], group, shallow=True)

# Make sure the new block size is reflected in the model parameters
new_file['model']['block_size'][()] = new_bs
old_solution = old_file['state']['solution']
new_solution = new_file['state']['solution']

for block in old_solution:
    old_solution_block = old_solution[block]
    new_solution_block = new_solution.require_group(block)
    old_conserved = old_solution_block['conserved']
    new_conserved = new_solution_block.create_dataset('conserved', (new_bs, new_bs), dtype=old_conserved.dtype)

    u = old_conserved[()]
    new_conserved[0::2, 0::2] = u
    new_conserved[0::2, 1::2] = u
    new_conserved[1::2, 0::2] = u
    new_conserved[1::2, 1::2] = u

    new_solution_block.copy(old_solution_block['integrated_source_terms'], 'integrated_source_terms')
