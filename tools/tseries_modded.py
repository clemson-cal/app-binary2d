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


# This function takes in a time series file and crunches out a tuple, where
# the first 3 variables are semi-major axis change due to accretion, gravity,
# and both of them combined. The fourth element is the time variable t
# associated with this particular run.
def ts_file_parse(fname):
    skip = 1
    h5f = h5py.File(fname, 'r')
    i = np.where(h5f['time_series']['time'] / 2 / np.pi < 200)[0][-1]

    tme = h5f['time_series']['time'][i::skip]
    m1 = h5f['time_series']['orbital_elements_change']['sink1']['1'][i::skip]
    m2 = h5f['time_series']['orbital_elements_change']['sink2']['1'][i::skip]
    a_sink_1 = h5f['time_series']['orbital_elements_change']['sink1']['0'][
               i::skip]
    a_sink_2 = h5f['time_series']['orbital_elements_change']['sink2']['0'][
               i::skip]
    a_grav_1 = h5f['time_series']['orbital_elements_change']['grav1']['0'][
               i::skip]
    a_grav_2 = h5f['time_series']['orbital_elements_change']['grav2']['0'][
               i::skip]

    orbit = tme / 2.0 / np.pi
    m = m1 + m2
    a_sink = a_sink_1 + a_sink_2
    a_grav = a_grav_1 + a_grav_2
    mean_mdot = td(m) / td(tme)

    adot_acc = smooth(np.diff(a_sink) / np.diff(tme)) / mean_mdot
    adot_grac = smooth(np.diff(a_grav) / np.diff(tme)) / mean_mdot
    adot_total = smooth(np.diff(a_sink + a_grav) / np.diff(tme)) / mean_mdot

    h5f.close()
    return adot_acc, adot_grac, adot_total, orbit


parser = argparse.ArgumentParser()
parser.add_argument("filename", nargs='*')
args = parser.parse_args()

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

count = 1
for name in args.filename:
    a1, a2, a3, t = ts_file_parse(name)
    name_list = name.split('/')
    act_name = name_list[len(name_list) - 1]

    label_name_grac = act_name + " a_grac" + str(count)
    label_name_accr = act_name + " a_accr" + str(count)
    label_name_totl = act_name + " a_totl" + str(count)
    ax1.plot(t[1:], a1, label=label_name_accr)
    ax2.plot(t[1:], a2, label=label_name_grac)
    # ax2.plot(t[1:], a3, label=label_name_totl)
    ax1.legend()
    ax2.legend()
    count += 1

plt.suptitle('app-binary2d disk mass sensitivity analysis')
plt.xlabel('Number of Orbits')
plt.ylabel(r'$\frac{\dot a}{\dot M_{total}}$', loc='top').set_rotation(0)
plt.show()
