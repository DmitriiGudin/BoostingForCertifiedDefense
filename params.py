# CIFAR-10 has 10 classes (0, 1, ..., 9), and each image is an array of 3072 integer values from 0 to 255, 1024 per channel (red, green, blue). 
# MNIST has 10 classes (0, 1, ..., 9), and each image is an array of 784 (?) values (?)
# Note that in this program labels are always 0, 1, 2, ... (integers). If you have other types of labels (i.e. strings), you have to code them with integers.

# GENERAL SETTINGS
model_filename = 'models/MNIST_50000_dep10.sav' # Name of the file containing the trained model


# DATA
training_file = 'data/MNIST_training_toy.hdf5' # Training hdf5 file location. 
testing_file = 'data/MNIST_testing_toy.hdf5' # Testing hdf5 file location. 
smoothed_training_file = 'data/MNIST_smoothed_training.hdf5' # File containing results of the smoothed testing on training data.
smoothed_testing_file = 'data/MNIST_smoothed_testing.hdf5' # File containing results of the smoothed testing on testing data.
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # List of classes corresponding to indeces 0, 1, ...
attacked_training_file = 'data/MNIST_attacked_training.hdf5' # File containing results of the attack on the training data.
attacked_testing_file = 'data/MNIST_attacked_testing.hdf5' # File containing results of the attack on the testing data.


# ADABOOST PARAMETERS
n_estimators = 50
DecisionTreeClassifier_max_depth = 10


# ADVERSARIAL ATTACK PARAMETERS
overshoot = 0.01 # Fraction to add to the DeepFool's perturbation
max_iter = 10 # Maximum number of iterations in the DeepFool's algorithm


# SMOOTHING PARAMETERS
gauss = 0.02 # Relative Gaussian noise variance
N_MC = 100 # Number of Monte Carlo runs for each image