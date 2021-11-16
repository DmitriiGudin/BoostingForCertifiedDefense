# CIFAR-10 has 10 classes (0, 1, ..., 9), and each image is an array of 3072 integer values from 0 to 255, 1024 per channel (red, green, blue). 
# MNIST has 10 classes (0, 1, ..., 9), and each image is an array of 784 (?) values (?)
# Note that in this program labels are always 0, 1, 2, ... (integers). If you have other types of labels (i.e. strings), you have to code them with integers.


# GENERAL SETTINGS
base_model_filename = 'models/MNIST_60000.sav' # Name of the file containing the base trained model
smooth_model_filename = 'models/MNIST_60000_smooth.sav' # Name of the file containing the smoothed trained model


# DATA
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # List of classes corresponding to indeces 0, 1, ...
N_features = 28*28 # Number of elements in each input array.
input_shape = (28, 28, 1) # Shape of one input for the classifier


# BASE XGBOOST MODEL PARAMETERS
num_boost_round = 250


# SMOOTHING PARAMETERS
gauss = 0.02 # Relative Gaussian noise variance
N_MC = 100 # Number of Monte Carlo runs for each image