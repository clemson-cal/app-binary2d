#!/bin/bash
echo "Loading relevant modules for binary2d suite."

module load rust/1.44.0-gcc/8.3.1
module load hdf5/1.10.6-gcc/8.3.1-cuda10_2-mpi
export HDF5_DIR=/software/spackages/linux-centos8-x86_64/gcc-8.3.1/hdf5-1.10.6-b2t32efolimdau6vqdmgnshwunb5hhyh

echo "Setup process complete."
echo "Note: source this file everytime a new node is in use."
