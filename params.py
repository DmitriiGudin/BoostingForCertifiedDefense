# GENERAL SETTINGS
model_filename = 'MNIST_50000_dep10.sav' # Name of the file containing the trained model


# DATA
dataset = 'MNIST' # 'CIFAR-10' or 'MNIST'. 
# CIFAR-10 has 10 classes (0, 1, ..., 9), and each image is an array of 3072 integer values from 0 to 255, 1024 per channel (red, green, blue). 
# MNIST has 10 classes (0, 1, ..., 9), and each image is an array of 784 (?) values (?)


# ADABOOST PARAMETERS
n_estimators=50


# ADVERSARIAL ATTACK PARAMETERS
overshoot = 0.02 # Fraction to add to the DeepFool's perturbation
max_iter = 10 # Maximum number of iterations in the DeepFool's algorithm


# SMOOTHING PARAMETERS
gauss = 0.02 # Relative Gaussian noise variance