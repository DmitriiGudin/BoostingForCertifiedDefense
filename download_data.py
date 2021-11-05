import numpy as np
import h5py
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    #X, Y = fetch_openml("CIFAR_10", version=1, return_X_y=True, as_frame=False, cache=True)
    #train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=50000, test_size=10000)
    #
    #hdf5_file = h5py.File('data/CIFAR_10_training.hdf5','w')
    #dset = hdf5_file.create_dataset('X', (50000,4096), dtype=int)
    #dset[:] = train_X[:]
    #dset = hdf5_file.create_dataset('X', (50000,), dtype=int)
    #dset[:] = train_Y[:]
    #hdf5_file.close()
    #
    #hdf5_file = h5py.File('data/CIFAR_10_testing.hdf5','w')
    #dset = hdf5_file.create_dataset('X', (10000,4096), dtype=int)
    #dset[:] = test_X[:]
    #dset = hdf5_file.create_dataset('X', (10000,), dtype=int)
    #dset[:] = test_Y[:]
    #hdf5_file.close()
    
    X, Y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, cache=True)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=1000, test_size=1000)
    
    hdf5_file = h5py.File('data/MNIST_training_toy.hdf5','w')
    dset = hdf5_file.create_dataset('X', (10000,784), dtype=int) #60,000
    dset[:] = train_X[:]
    dset = hdf5_file.create_dataset('Y', (10000,), dtype=int)
    for i in range(10000):
        dset[i] = int(train_Y[i])
    hdf5_file.close()
    
    hdf5_file = h5py.File('data/MNIST_testing_toy.hdf5','w')
    dset = hdf5_file.create_dataset('X', (10000,784), dtype=int) #10,000
    dset[:] = test_X[:]
    dset = hdf5_file.create_dataset('Y', (10000,), dtype=int)
    for i in range(10000):
        dset[i] = int(test_Y[i])
    hdf5_file.close()
    