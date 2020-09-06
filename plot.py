#!/usr/bin/env python3

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

h5f = h5py.File(args.filename, 'r')
rho = h5f["conserved"][...][:,:,0]

plt.imshow(rho)
plt.colorbar()
plt.show()
