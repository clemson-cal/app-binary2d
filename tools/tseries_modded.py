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
    e_sink_1 = h5f['time_series']['orbital_elements_change']['sink1']['3'][
               i::skip]
    e_sink_2 = h5f['time_series']['orbital_elements_change']['sink2']['3'][
               i::skip]
    e_grav_1 = h5f['time_series']['orbital_elements_change']['grav1']['3'][
               i::skip]
    e_grav_2 = h5f['time_series']['orbital_elements_change']['grav2']['3'][
               i::skip]

    orbit = tme / 2.0 / np.pi
    m = m1 + m2
    e_sink = e_sink_1 + e_sink_2
    e_grav = e_grav_1 + e_grav_2
    mean_mdot = td(m) / td(tme)

    edot_acc = smooth(np.diff(e_sink) / np.diff(tme)) / mean_mdot
    edot_grac = smooth(np.diff(e_grav) / np.diff(tme)) / mean_mdot
    edot_total = smooth(np.diff(e_sink + e_grav) / np.diff(tme)) / mean_mdot

    h5f.close()
    return edot_acc, edot_grac, edot_total, orbit


parser = argparse.ArgumentParser()
parser.add_argument("filename", nargs='*')
args = parser.parse_args()

fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
ax3 = fig.add_subplot(1, 1, 1)


for name in args.filename:
    e1, e2, e3, t = ts_file_parse(name)
    name_list = name.split('/')
    act_name = name_list[len(name_list) - 1]

    # label_name_grac = act_name + " e_grac" + str(count)
    # label_name_accr = act_name + " e_accr" + str(count)
    label_name_totl = act_name + " e_totl"
    # ax1.plot(t[1:], e1, label=label_name_accr)
    # ax2.plot(t[1:], e2, label=label_name_grac)
    ax3.plot(t[1:], e3, label=label_name_totl)
    # ax1.legend()
    # ax2.legend()
    ax3.legend()

plt.suptitle('binary orbit eccentricity, disk_radius = 10')
plt.xlabel('Number of Orbits')
plt.ylabel(r'$\frac{\dot e}{\dot M_{total}}$', fontsize=15).set_rotation(0)
plt.show()
