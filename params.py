# CIFAR-10 has 10 classes (0, 1, ..., 9), and each image is an array of 3072 integer values from 0 to 255, 1024 per channel (red, green, blue). 
# MNIST has 10 classes (0, 1, ..., 9), and each image is an array of 784 (?) values (?)

# GENERAL SETTINGS
model_filename = 'models/MNIST_50000_dep10.sav' # Name of the file containing the trained model


# DATA
training_file = 'data/MNIST_training_toy.hdf5' # Training hdf5 file location. 
testing_file = 'data/MNIST_testing_toy.hdf5' # Testing hdf5 file location. 
smoothed_training_file = 'data/MNIST_smoothed_training.hdf5' # File containing results of the smoothed testing on training data.
smoothed_testing_file = 'data/MNIST_smoothed_testing.hdf5' # File containing results of the smoothed testing on testing data.


# ADABOOST PARAMETERS
n_estimators = 50
DecisionTreeClassifier_max_depth = 3


# ADVERSARIAL ATTACK PARAMETERS
overshoot = 0.02 # Fraction to add to the DeepFool's perturbation
max_iter = 10 # Maximum number of iterations in the DeepFool's algorithm


# SMOOTHING PARAMETERS
gauss = 0.02 # Relative Gaussian noise variance
N_MC = 100 # Number of Monte Carlo runs for each image