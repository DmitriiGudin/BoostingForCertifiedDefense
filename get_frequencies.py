import numpy as np
import time
import h5py
import random
from art.utils import load_dataset
import params
import lib_train


if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    train_X, train_Y, test_X, test_Y, min_pixel_value, max_pixel_value = lib_train.get_data()
    hdf5_frequencies_XGBoost = h5py.File(params.hdf5_frequencies_XGBoost,'w')
    #dset = hdf5_frequencies_XGBoost.create_dataset('Accuracies', (len(params.certify_smoothing_sigma_list),len(params.certify_sigma_list)), float)
    
    for sigma in params.certify_smoothing_sigma_list:
    
        Print ("Sigma: " + str(sigma))
        train_X_noisy_training = lib_train.augment_images (train_X, sigma, min_pixel_value=min_pixel_value,  max_pixel_value=max_pixel_value)
        Print ("Training the model...")
        clf = lib_train.train_model (train_X_noisy_training, train_Y, test_X, test_Y, 0, min_pixel_value, max_pixel_value)
        Print ("Training complete!")
         
        dset = hdf5_frequencies_XGBoost.create_dataset('sigma_'+str(round(sigma,2)), (params.N_samples_frequencies,params.N_MC), float)
        indices = [i for i in range(0,len(test_X))]
        random.shuffle(indices)
        indices = indices[0:params.N_samples_frequencies]
        
        dset[:] = np.array([lib_train.clf_predict(clf, image, sigma, min_pixel_value, max_pixel_value, params.N_MC, full=True)[1] for image in test_X[indices]])
        
    hdf5_frequencies_XGBoost.close()