import numpy as np
import time
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import params


def train_model (train_X, train_Y): # Trains and returns an AdaBoost-SAMME model. train_X must be an array of 1-dimensional arrays, and train_Y is an array of integers corresponding to different labels.
    dtc = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    clf = AdaBoostClassifier(base_estimator=dtc, n_estimators=params.n_estimators, random_state=0)
    clf.fit(train_X, train_Y)
    return clf
    
    
def test_model (clf, test_X, test_Y): # Returns the accuracy (the fraction of correctly identified labels) of the model on the testing set; the arrays have the same structure as in the train_model function.
    return clf.score(test_X, test_Y)
    

if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    if params.dataset=='CIFAR-10': # CIFAR-10 parameters.
        X, Y = fetch_openml("CIFAR_10", version=1, return_X_y=True, as_frame=False, cache=True)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=50000, test_size=10000)
    elif params.dataset=='MNIST': # MNIST parameters.
        X, Y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, cache=True)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=60000, test_size=10000)
    else:
        Print ('Error: unfamiliar dataset specified in the params.py file.')
        quit()
    Print (str(min(train_X.flatten()))+', '+str(max(train_X.flatten()))) # Some diagnostic outputs
    Print (str(min(test_X.flatten()))+', '+str(max(test_X.flatten()))) #
    Print (str(min(train_Y.flatten()))+', '+str(max(train_Y.flatten()))) #
    Print (str(min(test_Y.flatten()))+', '+str(max(test_Y.flatten()))) #
    clf = train_model (train_X, train_Y) # Train the model.
    Print ('Training complete!')
    Print ('Accuracy on the training set: '+str(test_model (clf, train_X, train_Y)))
    Print ('Accuracy on the testing set: '+str(test_model (clf, test_X, test_Y)))
    pickle.dump(clf, open('models/'+params.model_filename, 'wb')) # Save the model to the models/ directory.