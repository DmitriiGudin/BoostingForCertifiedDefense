import autograd.numpy as np
from autograd import grad
import copy
from random import randrange
import params
import train


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
        Grad = grad(classify)(image)
        image += -(1+overshoot)*model.predict([image])[0]/np.linalg.norm(Grad)*Grad
    return image
 
 
def attacked_images (images, true_class, model, overshoot, max_iter, num_classes)
    array = []
    for image in images:
        array.append(attacked_image (image, true_class, model, overshoot, max_iter, num_classes))
    return np.array(array)

if __name__ == '__main__':
    print ('')