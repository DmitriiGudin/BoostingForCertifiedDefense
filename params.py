import numpy as np
# CIFAR-10 has 10 classes (0, 1, ..., 9), and each image is an array of 3072 integer values from 0 to 255, 1024 per channel (red, green, blue). 
# MNIST has 10 classes (0, 1, ..., 9), and each image is an array of 784 (?) values (?)
# Note that in this program labels are always 0, 1, 2, ... (integers). If you have other types of labels (i.e. strings), you have to code them with integers.
dataset= 'mnist'


# DATA
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # List of classes corresponding to indeces 0, 1, ...
N_features = 28*28 # Number of elements in each input array.
input_shape = (28, 28, 1) # Shape of one input for the classifier
hdf5_file = 'output/classes.hdf5'
hdf5_certify_XGBoost_accuracies = 'output/XGBoost_certify.hdf5'
hdf5_certify_Keras_accuracies = 'output/Keras_certify.hdf5'
hdf5_frequencies_XGBoost = 'output/XGBoost_frequencies.hdf5'
N_samples_frequencies = 1000 # How many images to consider when calculating frequencies.
worst_images_XGBoost = 'output/XGBoost_worst_images.hdf5'
#N_images_radius = 250 # How many "worst case scenario" images to select when calculating certified radii.
N_images_radius = 10
certified_defense_XGBoost = 'output/XGBoost_certified_defense.hdf5'


# BASE XGBOOST MODEL PARAMETERS
#num_boost_round = 200
num_boost_round = 200


# KERAS CNN MODEL PARAMETERS 
nb_epochs = 30


# SMOOTHING PARAMETERS
sigma = 0.2 # Relative Gaussian noise STD for tests
#N_MC = 100 # Number of Monte Carlo runs for each image
#N_MC = 1000
N_MC = 50
sigma_list = [0, 0.05, 0.1, 0.25, 0.5] # List of the Gaussian STD values for the full run
#certify_sigma_list = np.arange(0, 1+0.05, 0.05) # List of the Gaussian STD values used for certification 
certify_sigma_list = np.arange(0, 0.6+0.04, 0.04)
certify_smoothing_sigma_list = np.arange(0.2, 1+0.2, 0.2)


# ZOO ATTACK PARAMETERS 
#zoo_data_size = 500
zoo_data_size = 100