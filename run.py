import numpy as np
import time
import h5py
from art.utils import load_dataset
from art.attacks.evasion import ZooAttack
import params
import lib_train


if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    train_X, train_Y, test_X, test_Y, min_pixel_value, max_pixel_value = lib_train.get_data()
    hdf5_file = h5py.File(params.hdf5_file,'w')
    
    for sigma_training in params.sigma_list:
    
        train_X_noisy_training = lib_train.augment_images (train_X, sigma_training, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)
            
        Print ("")
        Print ("-----")
        Print ("Sigma smoothing: " + str(sigma_training))
        Print ("-----")
        Print ("Training the model...")
        clf = lib_train.train_model (train_X_noisy_training, train_Y, test_X, test_Y, 0, min_pixel_value, max_pixel_value)
        Print ("Training complete!")
            
        attack = ZooAttack(classifier=clf,confidence=0.5,targeted=False,learning_rate=1e-1,max_iter=50,binary_search_steps=50,initial_const=1e-1,abort_early=True,use_resize=False,use_importance=False,nb_parallel=500,batch_size=1,variable_h=0.01,)
        test_X_attacked = attack.generate(x=test_X[0:min(params.zoo_data_size,len(test_Y))])
        test_Y_attacked = test_Y[0:len(test_X_attacked)]
           
        for sigma_testing in params.sigma_list:
        
            Print ("Sigma testing: " + str(sigma_testing))
        
            train_X_noisy_testing = lib_train.augment_images (train_X, sigma_testing, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)
            test_X_noisy_testing = lib_train.augment_images (test_X, sigma_testing, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)
            
            Print ("Accuracy on the training set: " + str(lib_train.test_model(clf, train_X_noisy_testing, train_Y)))
            Print ("Accuracy on the testing set: " + str(lib_train.test_model(clf, test_X_noisy_testing, test_Y)))
            Print ("Randomized smoothing accuracy on the training set: " + str(lib_train.test_smooth_model (clf, train_X_noisy_testing, train_Y, params.sigma, min_pixel_value, max_pixel_value)))
            Print ("Randomized smoothing accuracy on the testing set: " + str(lib_train.test_smooth_model (clf, test_X_noisy_testing, test_Y, params.sigma, min_pixel_value, max_pixel_value)))
            
            Print ("Accuracy on the attacked testing set: " + str(lib_train.test_model(clf, test_X_attacked, test_Y_attacked)))
            Print ("Randomized smoothing accuracy on the attacked testing set: " + str(lib_train.test_smooth_model (clf, test_X_attacked, test_Y_attacked, params.sigma, min_pixel_value, max_pixel_value)))