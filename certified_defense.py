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
    certified_defense_XGBoost = h5py.File(params.certified_defense_XGBoost,'w')
    
    complete_fractions = []
    
    for sigma_model in params.certify_smoothing_sigma_list:
        train_X_noisy_training = lib_train.augment_images (train_X, sigma_model, min_pixel_value=min_pixel_value,  max_pixel_value=max_pixel_value)
        Print ("Training the model with the sigma = " + str(sigma_model) + "...")
        clf = lib_train.train_model (train_X_noisy_training, train_Y, test_X, test_Y, 0, min_pixel_value, max_pixel_value)
        Print ("Training complete!")
        Print ("")
        Print ("Obtaining the array of true predictions...")
        predictions = np.array([lib_train.clf_predict(clf, image, sigma=sigma_model, min_pixel_value=np.inf, max_pixel_value=np.inf, n=params.N_MC) for image in test_X])
        Print ("Done.")
        
        indices = np.arange(0, len(test_X))
        random.shuffle(indices)
        indices = indices[0:params.N_images_radius]
        
        all_fractions = []
        for sigma_noise in params.certify_sigma_list:
            Print ("Working on the noise level sigma = " + str(sigma_noise) + "...")
            fractions = []
            for i in range(params.N_images_radius):
                index = indices[i]
                prediction = predictions[index]
                image = test_X[index]
                augmented_images = np.array([lib_train.augment_image(image, sigma_noise, min_pixel_value, max_pixel_value) for i in range(params.N_MC)])
                results = np.array([lib_train.clf_predict(clf, img, sigma=sigma_model, min_pixel_value=np.inf, max_pixel_value=np.inf, n=params.N_MC) for img in augmented_images])
                fractions.append(len(results[results==prediction])/len(results))
            all_fractions.append(min(fractions))
            Print ("Done.")
        complete_fractions.append(all_fractions)
    complete_fractions = np.array(complete_fractions)
    
    dset = certified_defense_XGBoost.create_dataset('Fractions', (len(params.certify_smoothing_sigma_list),len(params.certify_sigma_list)), float)
    dset[:] = complete_fractions
    certified_defense_XGBoost.close()