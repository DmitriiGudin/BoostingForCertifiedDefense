import numpy as np
import xgboost as xgb
import tensorflow as tf
import inspect
import time
import pickle
import h5py
from art.estimators.classification import XGBoostClassifier
from art.estimators.certification.randomized_smoothing import RandomizedSmoothingMixin
from art.utils import load_mnist
import params


def train_base_model (train_X, train_Y, test_X, test_Y): # Trains and returns an XGBoost model. train_X must be an array of 1-dimensional arrays, and train_Y is an array of integers corresponding to different labels.
    parameters = {"objective": "multi:softprob", "metric": "accuracy", "num_class": len(params.classes)}
    dtrain = xgb.DMatrix(train_X, np.argmax(train_Y, axis=1))
    dtest = xgb.DMatrix(test_X, np.argmax(test_Y, axis=1))
    evals = [(dtest, "test"), (dtrain, "train")]
    model = xgb.train(params=parameters, dtrain=dtrain, num_boost_round=50, evals=evals)
    clf = XGBoostClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), nb_features=params.N_features, nb_classes=len(params.classes))
    return clf
    
    
def train_smooth_model (base_model, train_X, train_Y): # Trains and returns a randomized-smoothed model based on the specified base model. train_X must be an array of 1-dimensional arrays, and train_Y is an array of integers corresponding to different labels.
    wrapper = RandomizedSmoothingMixin(sample_size = params.N_MC, scale=params.gauss)
    print (base_model)
    print (wrapper)
    print (inspect.get_members(wrapper))
    clf = wrapper(base_model)
    clf.fit(train_X, train_Y)
    return clf
    
    
def test_model (clf, test_X, test_Y): # Returns the accuracy (the fraction of correctly identified labels) of the model on the testing set; the arrays have the same structure as in the train_model function.
    predictions = clf.predict(test_X)
    return np.sum(np.argmax(predictions, axis=1) == np.argmax(test_Y, axis=1)) / len(test_Y)
    

if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    (train_X, train_Y), (test_X, test_Y), min_pixel_value, max_pixel_value = load_mnist()
    nb_samples_train = train_X.shape[0]
    nb_samples_test = test_X.shape[0]
    train_X = train_X.reshape((nb_samples_train, params.N_features))
    test_X = test_X.reshape((nb_samples_test, params.N_features))
    
    clf_base = train_base_model (train_X, train_Y, test_X, test_Y) # Train the base model.
    Print ('Training of the base model complete!')
    Print ('Accuracy on the training set: '+str(test_model (clf_base, train_X, train_Y)))
    Print ('Accuracy on the testing set: '+str(test_model (clf_base, test_X, test_Y)))
    pickle.dump(clf_base, open(params.base_model_filename, 'wb')) # Save the model to the models/ directory.
    
    clf_smooth = train_smooth_model (clf_base, train_X, train_Y) # Train the randomized-smoothed model.
    Print ('Training of the randomized-smoothed model complete!')
    Print ('Accuracy on the training set: '+str(test_model (clf_smooth, train_X, train_Y)))
    Print ('Accuracy on the testing set: '+str(test_model (clf_smooth, test_X, test_Y)))
    pickle.dump(clf_smooth, open(params.smooth_model_filename, 'wb')) # Save the model to the models/ directory.