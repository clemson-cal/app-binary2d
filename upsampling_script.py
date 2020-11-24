"""
This script takes in a checkpoint file and up-scales each cell in each block
into four cells. The resulting resolution will increase by 4 times compared to
the older file.

Author: Zhongtian Hu, Clemson University @ 2020.
"""

import argparse
import numpy as np
import h5py

# Parse command line argument for filename.
default_fname = "ups_file"
parser = argparse.ArgumentParser()
parser.add_argument("input", help='The input filename of the h5 file.')
parser.add_argument("--output",
                    dest="output",
                    default=default_fname,
                    help="The output file name. "
                         "Default will be 'ups_file.h5'")
# parser.add_argument()
args = parser.parse_args()

# Reading the input file.
old_file_pointer = h5py.File(args.input, 'r')

# Check on the data stored.
old_bs = old_file_pointer['model']['block_size'][()]
new_bs = old_bs * 2
nb = old_file_pointer['model']['num_blocks'][()]

print("The checkpoint file provided is: " + args.input)
print("The output file will be named: " + args.output + ".h5")
print("The number of block is: ", end="")
print(nb)
print("The old block size is: ", end="")
print(old_bs)
print("The new block size will be: ", end="")
print(new_bs)

# Create a new h5 file with the filename and copy the content of the old file
# without the conserved data.
new_file_pointer = h5py.File(args.output, 'w-')

new_file_pointer.copy(old_file_pointer['model'], 'model')
new_file_pointer.copy(old_file_pointer['iteration'], 'iteration')
new_file_pointer.copy(old_file_pointer['tasks'], 'tasks')
new_file_pointer.copy(old_file_pointer['time'], 'time')
new_conserved = new_file_pointer.create_group('conserved')

# Change the old block size value in new file to the updated one.
new_file_pointer['model']['block_size'][()] = new_bs

# Using the subgroup names in conserved of the old file, create new groups
# with each containing updated cells.
for group in old_file_pointer['conserved']:
    # For each block, create a new dataset that has a dimension of
    # (old_bs * 2, old_bs * 2, 3)
    dtype = old_file_pointer['conserved'][group].dtype
    new_dataset = new_file_pointer['conserved'].create_dataset(group, (new_bs, new_bs), dtype=dtype)
    new_dataset[0::2, 0::2, ] = old_file_pointer['conserved'][group][()]
    new_dataset[0::2, 1::2, ] = old_file_pointer['conserved'][group][()]
    new_dataset[1::2, 0::2, ] = old_file_pointer['conserved'][group][()]
    new_dataset[1::2, 1::2, ] = old_file_pointer['conserved'][group][()]

# Code below is for testing only. Remove before publishing.
'''print("Test messages:")
for group in new_file_pointer['conserved']:
    print(type(new_file_pointer['conserved'][group]))
    print(new_file_pointer['conserved'][group])
'''

# Closing the output file.
new_file_pointer.close()

# Closing the input file.
old_file_pointer.close()
