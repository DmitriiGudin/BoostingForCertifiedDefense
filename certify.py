import numpy as np
import time
import h5py
from art.utils import load_dataset
import params
import lib_train


if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    train_X, train_Y, test_X, test_Y, min_pixel_value, max_pixel_value = lib_train.get_data()
    hdf5_certify_XGBoost_accuracies = h5py.File(params.hdf5_certify_XGBoost_accuracies,'w')
    dset = hdf5_certify_XGBoost_accuracies.create_dataset('Accuracies', (len(params.certify_smoothing_sigma_list),len(params.certify_sigma_list)), float)
    
    accuracies = np.zeros((len(params.certify_smoothing_sigma_list),len(params.certify_sigma_list)))
    
    for i, sigma_model in enumerate(params.certify_smoothing_sigma_list):
    
        Print ("")
        Print ("------")
        Print ("Sigma of model: " + str(sigma_model))
        Print ("------")
        train_X_noisy_training = lib_train.augment_images (train_X, sigma_model, min_pixel_value=min_pixel_value,  max_pixel_value=max_pixel_value)
        Print ("Training the model...")
        clf = lib_train.train_model (train_X_noisy_training, train_Y, test_X, test_Y, 0, min_pixel_value, max_pixel_value)
        Print ("Training complete!")
        for j, sigma_noise in enumerate(params.certify_sigma_list):
    
            test_X_noisy_testing = lib_train.augment_images (test_X, sigma_noise, min_pixel_value=min_pixel_value,  max_pixel_value=max_pixel_value)
            
            Print ("")
            Print ("Sigma of noise: " + str(sigma_noise))
         
            if sigma_model > 0:
                acc = lib_train.test_smooth_model (clf, test_X_noisy_testing, test_Y, sigma_model, min_pixel_value, max_pixel_value)
            else:
                acc = lib_train.test_model (clf, test_X_noisy_testing, test_Y)
            Print ("Accuracy on the testing set: " + str(acc))
            accuracies[i][j] = acc 
    dset[:] = accuracies
    hdf5_certify_XGBoost_accuracies.close()