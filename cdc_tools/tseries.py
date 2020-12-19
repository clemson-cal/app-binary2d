#!/usr/bin/env python3

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def smooth(x):
    return savgol_filter(x, 101, 3)


def td(x):
    return x[-1] - x[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    skip = 1

    h5f  = h5py.File(args.filename, 'r')
    i = np.where(h5f['time_series']['time'] / 2 / np.pi < 200)[0][-1]

    t        = h5f['time_series']['time'][i::skip]
    M1       = h5f['time_series']['orbital_elements_change']['sink1']['1'][i::skip]
    M2       = h5f['time_series']['orbital_elements_change']['sink2']['1'][i::skip]
    a_sink_1 = h5f['time_series']['orbital_elements_change']['sink1']['0'][i::skip]
    a_sink_2 = h5f['time_series']['orbital_elements_change']['sink2']['0'][i::skip]
    a_grav_1 = h5f['time_series']['orbital_elements_change']['grav1']['0'][i::skip]
    a_grav_2 = h5f['time_series']['orbital_elements_change']['grav2']['0'][i::skip]

    orbit = t / 2.0 / np.pi
    M = M1 + M2
    a_sink = a_sink_1 + a_sink_2
    a_grav = a_grav_1 + a_grav_2
    mean_mdot = td(M) / td(t)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(orbit[1:], smooth(np.diff(a_sink)          / np.diff(t)) / mean_mdot, label=r'$\dot a_{\rm acc}$')
    ax1.plot(orbit[1:], smooth(np.diff(a_grav)          / np.diff(t)) / mean_mdot, label=r'$\dot a_{\rm grac}$')
    ax1.plot(orbit[1:], smooth(np.diff(a_sink + a_grav) / np.diff(t)) / mean_mdot, label=r'$\dot a_{\rm tot}$')
    ax1.legend()

    plt.show()


if __name__ == "__main__":
    main()
