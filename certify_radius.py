import numpy as np
import time
import h5py
import random
from scipy import stats
from art.utils import load_dataset
import params
import lib_train


if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    train_X, train_Y, test_X, test_Y, min_pixel_value, max_pixel_value = lib_train.get_data()
    train_X_noisy_training = lib_train.augment_images (train_X, params.sigma, min_pixel_value=min_pixel_value,  max_pixel_value=max_pixel_value)
    worst_images_XGBoost = h5py.File(params.worst_images_XGBoost,'w')
    
    Print ("Training the model with the base sigma " + str(params.sigma) + "...")
    clf = lib_train.train_model (train_X_noisy_training, train_Y, test_X, test_Y, 0, min_pixel_value, max_pixel_value)
    Print ("Training complete!")
    
    indices = [i for i in range(len(test_X))]
    random.shuffle(indices)
    indices = [i for i in range(params.N_images_radius)]
    
    for sigma in params.certify_sigma_list:
    
        dset = worst_images_XGBoost.create_dataset('sigma_'+str(round(sigma,2)), (params.N_MC,), float)
    
        Print ("Sigma: " + str(sigma))
        
        accuracies = []
        for i in range(params.N_MC):
            test_X_noisy_testing = lib_train.augment_images (test_X[indices], sigma, min_pixel_value=min_pixel_value,  max_pixel_value=max_pixel_value)
            accuracies.append (lib_train.test_smooth_model (clf, test_X_noisy_testing, test_Y[indices], params.sigma, min_pixel_value, max_pixel_value))
        Print ("Worst case accuracy: "+str(min(accuracies)))
        dset[:] = accuracies