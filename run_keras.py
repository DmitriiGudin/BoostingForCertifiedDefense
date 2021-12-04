import numpy as np
import time
import h5py
from art.utils import load_dataset
from art.attacks.evasion import ZooAttack
import tensorflow as tf
import params
import lib_train


if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    train_X, train_Y, test_X, test_Y, min_pixel_value, max_pixel_value = lib_train.get_data()
    hdf5_file = h5py.File(params.hdf5_file,'w')
    
    for sigma in params.sigma_list:
    
        train_X_noisy_training = lib_train.augment_images (train_X, sigma, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)
            
        Print ("")
        Print ("-----")
        Print ("Sigma of injected noise: " + str(sigma))
        Print ("-----")
        Print ("Training the model...")
        clf = lib_train.train_keras_model (train_X_noisy_training, train_Y, test_X, test_Y, 0, min_pixel_value, max_pixel_value)
        Print ("Training complete!")
            
        attack = ZooAttack(classifier=clf,confidence=0.5,targeted=False,learning_rate=1e-1,max_iter=50,binary_search_steps=50,initial_const=1e-1,abort_early=True,use_resize=False,use_importance=False,nb_parallel=500,batch_size=1,variable_h=0.01,)
        attack_X_reshaped = test_X[0:min(params.zoo_data_size,len(test_Y))]
        attack_X_reshaped = attack_X_reshaped.reshape(len(attack_X_reshaped),len(attack_X_reshaped[0]),1)
        test_X_attacked = attack.generate(x=attack_X_reshaped)
        test_X_attacked = test_X_attacked.reshape(len(test_X_attacked),len(test_X_attacked[0]))
        test_Y_attacked = test_Y[0:len(test_X_attacked)]
           
        train_X_noisy_testing = lib_train.augment_images (train_X, sigma_testing, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)
        test_X_noisy_testing = lib_train.augment_images (test_X, sigma_testing, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)
            
        Print ("Accuracy on the training set: " + str(lib_train.test_model(clf, train_X_noisy_testing, train_Y, isKeras=True)))
        Print ("Accuracy on the testing set: " + str(lib_train.test_model(clf, test_X_noisy_testing, test_Y, isKeras=True)))
        Print ("Randomized smoothing accuracy on the training set: " + str(lib_train.test_smooth_model (clf, train_X_noisy_testing, train_Y, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))
        Print ("Randomized smoothing accuracy on the testing set: " + str(lib_train.test_smooth_model (clf, test_X_noisy_testing, test_Y, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))
            
        Print ("Accuracy on the attacked testing set: " + str(lib_train.test_model(clf, test_X_attacked, test_Y_attacked, isKeras=True)))
        Print ("Randomized smoothing accuracy on the attacked testing set: " + str(lib_train.test_smooth_model (clf, test_X_attacked, test_Y_attacked, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))