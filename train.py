import numpy as np
import time
import pickle
import h5py
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import params


def train_model (train_X, train_Y): # Trains and returns an AdaBoost-SAMME model. train_X must be an array of 1-dimensional arrays, and train_Y is an array of integers corresponding to different labels.
    dtc = DecisionTreeClassifier(criterion="entropy", max_depth=params.DecisionTreeClassifier_max_depth)
    clf = AdaBoostClassifier(base_estimator=dtc, n_estimators=params.n_estimators, random_state=0)
    clf.fit(train_X, train_Y)
    return clf
    
    
def test_model (clf, test_X, test_Y): # Returns the accuracy (the fraction of correctly identified labels) of the model on the testing set; the arrays have the same structure as in the train_model function.
    return clf.score(test_X, test_Y)
    

if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    hdf5_file = h5py.File(params.training_file,'r')
    train_X, train_Y = hdf5_file['X'][:], hdf5_file['Y'][:]
    hdf5_file.close()
    hdf5_file = h5py.File(params.testing_file,'r')
    test_X, test_Y = hdf5_file['X'][:], hdf5_file['Y'][:]
    hdf5_file.close()
    
    clf = train_model (train_X, train_Y) # Train the model.
    Print ('Training complete!')
    Print ('Accuracy on the training set: '+str(test_model (clf, train_X, train_Y)))
    Print ('Accuracy on the testing set: '+str(test_model (clf, test_X, test_Y)))
    pickle.dump(clf, open(params.model_filename, 'wb')) # Save the model to the models/ directory.