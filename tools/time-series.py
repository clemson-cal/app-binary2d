#!/usr/bin/env python3

"""
Tool to extract time series data from the binary code's
CBOR-formatted checkpoint files, and convert them to a
pickle for faster loading.
"""

import argparse
import pickle
import cdc_loader


def getpath(d, path):
    for part in path:
        d = d[part]
    return d


def setpath(d, path, value):
    p = path[0]
    if len(path) == 1:
        d[p] = value
    else:
        if p not in d:
            d[p] = dict()
        setpath(d[p], path[1:], value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='time-series.py path/to/chkpt.1234.cbor',
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('filename')
    parser.add_argument('--output', metavar='time_series.pk', default=None, type=str, help='output filename')
    args = parser.parse_args()

    print(f'load {args.filename}')
    app = cdc_loader.app(args.filename)
    time_series = app.time_series    

    if not time_series:
        print('warning: time series data was empty')
        exit()

    paths = []
    for key in time_series[0]:
        try:
            for subkey in time_series[0][key]:
                paths.append((key, subkey))
        except TypeError:
            paths.append((key,))

    result = dict()
    for path in paths:
        series = [getpath(sample, path) for sample in time_series]
        setpath(result, path, series)
    if args.output is None:
        fname = args.filename.replace('chkpt', 'time_series').replace('.cbor', '.pk')
    else:
        fname = args.output

    print(f'save {fname}')
    pickle.dump(result, open(fname, 'wb'))
