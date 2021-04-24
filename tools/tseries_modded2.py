import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

if __name__ == "__main__":

    data = []
    with (open("/Users/jackhu/Google_Drive/Rust_personal/projects/app"
               "-binary2d/data/eccentricity_studies/mach_number_studies/mn10"
               "/e0.04/time_series.0070.pk", "rb")) as openfile:
        while True:
            try:
                data.append(pickle.load(openfile))
            except EOFError:
                break

    print(data)
