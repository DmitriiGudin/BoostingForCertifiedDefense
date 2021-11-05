import numpy as np
import numdifftools as nd
import copy
import pickle
import h5py
from random import randrange
import params
import train


datafile = params.training_file
output_file = params.attacked_training_file


def attacked_image (image, true_class, model, overshoot, max_iter, num_classes):
    def classify (image):
        return model.predict([image])[0]
    pert_image = copy.deepcopy(image)
    new_class = true_class 
    while (new_class == true_class):
        new_class = randrange(num_classes)
    i, label = 0, true_class
    while i<max_iter and label==true_class:
        pert = np.inf
        Grad = nd.Gradient(classify)(image.astype(float))
        if (float(np.linalg.norm(Grad)) != 0):
            image += Grad*(-float(1+overshoot)*float(model.predict([image])[0])/float(np.linalg.norm(Grad)))
    return image
 
 
def attacked_images (images, true_class, model, overshoot, max_iter, num_classes):
    array = []
    for image in images:
        array.append(attacked_image (image, true_class, model, overshoot, max_iter, num_classes))
    return np.array(array)

if __name__ == '__main__':
    with open(params.model_filename, 'rb') as f:
        clf = pickle.load(f)
        
    hdf5_datafile = h5py.File(datafile, 'r')    
    X_true = hdf5_datafile['X'][:]
    Y_true = hdf5_datafile['Y'][:]
    X_attacked = np.array([attacked_image(X_true[i], Y_true[i], clf, params.overshoot, params.max_iter, len(params.classes)) for i in range(len(Y_true))])
    hdf5_file = h5py.File(output_file,'w')
    dset = hdf5_file.create_+dataset('X', (len(Y_true),), dtype=int)
    dset[:] = X_attacked 
    hdf5_file.close()