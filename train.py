import numpy as np
import xgboost as xgb
import tensorflow as tf
import inspect
import time
import pickle
import h5py
from scipy import stats
from art.estimators.classification import XGBoostClassifier
from art.utils import load_dataset
import params


def train_base_model (train_X, train_Y, test_X, test_Y): # Trains and returns an XGBoost model. train_X must be an array of 1-dimensional arrays, and train_Y is an array of integers corresponding to different labels.
    parameters = {"objective": "multi:softprob", "metric": "accuracy", "num_class": len(params.classes)}
    dtrain = xgb.DMatrix(train_X, np.argmax(train_Y, axis=1))
    dtest = xgb.DMatrix(test_X, np.argmax(test_Y, axis=1))
    evals = [(dtest, "test"), (dtrain, "train")]
    model = xgb.train(params=parameters, dtrain=dtrain, num_boost_round=params.num_boost_round, evals=evals)
    clf = XGBoostClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), nb_features=params.N_features, nb_classes=len(params.classes))
    return clf
    
    
def clf_smooth_predict (clf, image, min_pixel_value, max_pixel_value, full=False):
    preds = []
    for i in range(params.N_MC):
        image_noisy = image + np.random.normal(0,np.sqrt(params.gauss)*(max_pixel_value-min_pixel_value),(len(image),))
        image_noisy = np.clip(image_noisy, min_pixel_value, max_pixel_value)
        prediction = clf.predict(np.array([image_noisy]))
        preds.append(np.argmax(prediction))
    preds = np.array(preds)
    if full:
        return stats.mode(preds), preds
    return stats.mode(preds)[0][0]
    
    
def test_model (clf, test_X, test_Y): # Returns the accuracy (the fraction of correctly identified labels) of the model on the testing set; the arrays have the same structure as in the train_model function.
    predictions = clf.predict(test_X)
    return np.sum(np.argmax(predictions, axis=1) == np.argmax(test_Y, axis=1)) / len(test_Y)
    
    
def test_smooth_model (clf, test_X, test_Y, min_pixel_value, max_pixel_value, full=False):
    predictions = np.array([clf_smooth_predict(clf, x, min_pixel_value, max_pixel_value, full) for x in test_X])
    return np.sum(np.argmax(test_Y, axis=1) == predictions) / len(test_Y)
    

if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    (train_X, train_Y), (test_X, test_Y), min_pixel_value, max_pixel_value = load_dataset('mnist')
    nb_samples_train = train_X.shape[0]
    nb_samples_test = test_X.shape[0]
    train_X = train_X.reshape((nb_samples_train, params.N_features))
    test_X = test_X.reshape((nb_samples_test, params.N_features))
    
    Print ('Training on the original set...')
    clf = train_base_model (train_X, train_Y, test_X, test_Y) # Train the base model on the original set.
    Print ('Training of the base model complete!')
    Print ('Accuracy on the training set: '+str(test_model (clf, train_X, train_Y)))
    Print ('Accuracy on the testing set: '+str(test_model (clf, test_X, test_Y)))
    pickle.dump(clf, open(params.base_model_filename, 'wb')) # Save the model to the models/ directory.
    Print ('Accuracy of the randomized-smoothed model on the original training set: '+str(test_smooth_model (clf, train_X, train_Y, min_pixel_value, max_pixel_value)))
    Print ('Accuracy of the randomized-smoothed model on the original testing set: '+str(test_smooth_model (clf, test_X, test_Y, min_pixel_value, max_pixel_value)))
    
    train_X_noisy = train_X + np.random.normal(0,np.sqrt(params.gauss)*(max_pixel_value-min_pixel_value),train_X.shape)
    train_X_noisy = np.clip(train_X_noisy, min_pixel_value, max_pixel_value)
    
    Print ('Training on the noisy set...')
    clf_smoothed = train_base_model (train_X_noisy, train_Y, test_X, test_Y) # Train the base model on the noisy set.
    Print ('Training of the base model complete!')
    Print ('Accuracy on the training set: '+str(test_model (clf_smoothed, train_X, train_Y)))
    Print ('Accuracy on the testing set: '+str(test_model (clf_smoothed, test_X, test_Y)))
    pickle.dump(clf_smoothed, open(params.smooth_model_filename, 'wb')) # Save the model to the models/ directory.
    
    train_X_noisy = train_X + np.random.normal(0,np.sqrt(params.gauss)*(max_pixel_value-min_pixel_value),train_X.shape)
    train_X_noisy = np.clip(train_X_noisy, min_pixel_value, max_pixel_value)
    
    Print ('Accuracy of the randomized-smoothed model on the noisy training set: '+str(test_smooth_model (clf_smoothed, train_X, train_Y, min_pixel_value, max_pixel_value)))
    Print ('Accuracy of the randomized-smoothed model on the noisy testing set: '+str(test_smooth_model (clf_smoothed, test_X, test_Y, min_pixel_value, max_pixel_value)))