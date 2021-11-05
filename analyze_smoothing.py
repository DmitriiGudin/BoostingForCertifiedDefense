import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import params


smoothed_file = params.smoothed_testing_file
datafile = params.testing_file


if __name__ == '__main__':
    with open(params.model_filename, 'rb') as f:
        clf = pickle.load(f)

    hdf5_file = h5py.File(smoothed_file, 'r')
    hdf5_datafile = h5py.File(datafile, 'r')
    Y_smoothed = hdf5_file['Y'][:]
    Y = hdf5_datafile['Y'][:]