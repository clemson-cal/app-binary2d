#!/usr/bin/env python3

import argparse
import h5py
import numpy as np

def rust_string(value):
    if type(value) is bytes:
        return value.decode('utf-8')
    elif type(value) is np.bool_:
        return str(value).lower()
    return value

parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='+')
args = parser.parse_args()

for filename in args.filenames:
    print(' '.join('{}={}'.format(key, rust_string(val[()])) for key, val in h5py.File(filename, 'r')['model'].items()))
