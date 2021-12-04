import numpy as np
import xgboost as xgb
import time
from scipy import stats
from art.estimators.classification import XGBoostClassifier, KerasClassifier
from art.utils import load_dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import params


def get_data (dataset='mnist', N_features=params.N_features):
    (train_X, train_Y), (test_X, test_Y), min_pixel_value, max_pixel_value = load_dataset(dataset)
    nb_samples_train = train_X.shape[0]
    nb_samples_test = test_X.shape[0]
    train_X = train_X.reshape((nb_samples_train, N_features))
    test_X = test_X.reshape((nb_samples_test, N_features))
    return (train_X, train_Y, test_X, test_Y, min_pixel_value, max_pixel_value)


def augment_image (image, sigma, min_pixel_value=np.inf, max_pixel_value=np.inf): # Augments the supplied image (a 1-dimensional numpy array) with the gaussian noise of the relative level sigma. min_pixel_value and max_pixel_value are used for calculating the absolute value of noise and for clipping of the augmented image. min_pixel_value and max_pixel_value, if unspecified, are calculated from the supplied image.
    if min_pixel_value==np.inf or max_pixel_value==np.inf:
        min_pixel_value = min(image)
        max_pixel_value = max(image)
    image_noisy = image + np.random.normal(0,sigma*(max_pixel_value-min_pixel_value),(len(image),))
    image_noisy = np.clip(image_noisy, min_pixel_value, max_pixel_value)
    return image_noisy
    
    
def augment_images (images, sigma, min_pixel_value=np.inf, max_pixel_value=np.inf): # Augments the supplied images (a 2-dimensional numpy array) with the gaussian noise of the relative level sigma. min_pixel_value and max_pixel_value are used for calculating the absolute value of noise and for clipping of the augmented image. min_pixel_value and max_pixel_value, if unspecified, are calculated from the supplied data.
    if min_pixel_value==np.inf or max_pixel_value==np.inf:
        min_pixel_value = min(images.flatten)
        max_pixel_value = max(images.flatten)
    return np.array([augment_image(i, sigma, min_pixel_value, max_pixel_value) for i in images])
    

def train_model (train_X, train_Y, test_X, test_Y, sigma=0, min_pixel_value=np.inf, max_pixel_value=np.inf): # Trains and returns an XGBoost model. train_X must be an array of 1-dimensional arrays, and train_Y is an array of integers corresponding to different labels. test_X and test_Y must have the same structure. sigma is the relative Gaussian noise level to apply to images; if 0 or negative, then the training is performed on the set of the original non-augmented images. min_pixel_value and max_pixel_value, if unspecified, are calculated from the supplied data.
    parameters = {"objective": "multi:softprob", "metric": "accuracy", "num_class": len(params.classes)}
    if min_pixel_value==np.inf or max_pixel_value==np.inf:
        min_pixel_value = min(train_X.flatten)
        max_pixel_value = max(train_X.flatten)
    if sigma>0:
        train_X = augment_images (train_X, sigma, min_pixel_value, max_pixel_value)
    dtrain = xgb.DMatrix(train_X, np.argmax(train_Y, axis=1))
    dtest = xgb.DMatrix(test_X, np.argmax(test_Y, axis=1))
    evals = [(dtest, "test"), (dtrain, "train")]
    model = xgb.train(params=parameters, dtrain=dtrain, num_boost_round=params.num_boost_round, evals=evals)
    clf = XGBoostClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), nb_features=params.N_features, nb_classes=len(params.classes))
    return clf
    
    
def train_keras_model (train_X, train_Y, test_X, test_Y, sigma=0, min_pixel_value=np.inf, max_pixel_value=np.inf): # Trains and returns a Keras CNN model. train_X must be an array of 1-dimensional arrays, and train_Y is an array of integers corresponding to different labels. test_X and test_Y must have the same structure. sigma is the relative Gaussian noise level to apply to images; if 0 or negative, then the training is performed on the set of the original non-augmented images. min_pixel_value and max_pixel_value, if unspecified, are calculated from the supplied data.
    if min_pixel_value==np.inf or max_pixel_value==np.inf:
        min_pixel_value = min(train_X.flatten)
        max_pixel_value = max(train_X.flatten)
    if sigma>0:
        train_X = augment_images (train_X, sigma, min_pixel_value, max_pixel_value)

    model = Sequential()
    model.add(Conv1D(filters=4, kernel_size=5, strides=1, activation="relu", input_shape=(len(train_X[0]), 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=10, kernel_size=5, strides=1, activation="relu", input_shape=(len(train_X[0])-5, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])
    clf = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
    clf.fit(train_X.reshape((len(train_X), len(train_X[0]), 1)), train_Y, batch_size=64, nb_epochs=params.nb_epochs)
    return clf
    
    
def clf_predict (clf, image, sigma=0, min_pixel_value=np.inf, max_pixel_value=np.inf, n=params.N_MC, full=False, isKeras=False): # Applies randomized smoothing to the pretrained classifier clf on the image if sigma>0, otherwise uses the classifier in a standard way, to predict the image's label. min_pixel_value and max_pixel_value, if unspecified, are calculated from the supplied image. In case of randomized smoothing, full=True returns not only the prediction, but also the array of the individual predictions of N_MC noisy images.
    preds = []
    if sigma>0:
        for i in range(n):
            image_noisy = augment_image(image, sigma, min_pixel_value, max_pixel_value)
            if isKeras:
                image_noisy = image_noisy.reshape(len(image), 1)
            prediction = clf.predict(np.array([image_noisy]))[0]
            preds.append(np.argmax(prediction))
        preds = np.array(preds)
        if full:
            return stats.mode(preds)[0][0], preds
        else:
            return stats.mode(preds)[0][0]
    else:
        if isKeras:
            image = image.reshape(len(image), 1)
        prediction = clf.predict(np.array([image]))[0]
        return prediction
    
    
def test_model (clf, test_X, test_Y, isKeras=False): # Returns the accuracy (the fraction of correctly identified labels) of the model on the testing set; the arrays have the same structure as in the train_model function.
    if isKeras:
        test_X = test_X.reshape((len(test_X), len(test_X[0]), 1))
    predictions = clf.predict(test_X)
    return np.sum(np.argmax(predictions, axis=1) == np.argmax(test_Y, axis=1)) / len(test_Y)
    
    
def test_smooth_model (clf, test_X, test_Y, sigma, min_pixel_value, max_pixel_value, n=params.N_MC, full=False, isKeras=False):
    predictions = np.array([clf_predict(clf, x, sigma, min_pixel_value, max_pixel_value, n, full, isKeras) for x in test_X])
    return np.sum(np.argmax(test_Y, axis=1) == predictions) / len(test_Y)
    

if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    train_X, train_Y, test_X, test_Y, min_pixel_value, max_pixel_value = get_data()
    train_X_noisy_training = augment_images (train_X, params.sigma, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)
    train_X_noisy_testing = augment_images (train_X, params.sigma, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)
    test_X_noisy_testing = augment_images (test_X, params.sigma, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)
    
    #Print ("Training the model on the original set...")
    #clf = train_model (train_X, train_Y, test_X, test_Y, 0, min_pixel_value, max_pixel_value)
    #Print ("Training complete!")
    #Print ("Training the model on the augmented set...")
    #noisy_clf = train_model (train_X_noisy_training, train_Y, test_X, test_Y, 0, min_pixel_value, max_pixel_value)
    #Print ("Training complete!")
    
    #Print ("")
    #Print ("----------")
    #Print ("ORIGINAL MODEL")
    #Print ("----------")
    #Print ("Accuracy on the original training set: " + str(test_model(clf, train_X, train_Y)))
    #Print ("Accuracy on the original testing set: " + str(test_model(clf, test_X, test_Y)))
    #Print ("Accuracy on the augmented training set: " + str(test_model(clf, train_X_noisy_testing, train_Y)))
    #Print ("Accuracy on the augmented testing set: " + str(test_model(clf, test_X_noisy_testing, test_Y)))
    #Print ("--------")
    #Print ("Randomized smoothing accuracy on the original training set: " + str(test_smooth_model (clf, train_X, train_Y, params.sigma, min_pixel_value, max_pixel_value)))
    #Print ("Randomized smoothing accuracy on the original testing set: " + str(test_smooth_model (clf, test_X, test_Y, params.sigma, min_pixel_value, max_pixel_value)))
    #Print ("Randomized smoothing accuracy on the augmented training set: " + str(test_smooth_model (clf, train_X_noisy_testing, train_Y, params.sigma, min_pixel_value, max_pixel_value)))
    #Print ("Randomized smoothing accuracy on the augmented testing set: " + str(test_smooth_model (clf, test_X_noisy_testing, test_Y, params.sigma, min_pixel_value, max_pixel_value)))
    #Print ("")
    #Print ("")
    #Print ("----------")
    #Print ("AUGMENTED MODEL")
    #Print ("----------")
    #Print ("Accuracy on the original training set: " + str(test_model(noisy_clf, train_X, train_Y)))
    #Print ("Accuracy on the original testing set: " + str(test_model(noisy_clf, test_X, test_Y)))
    #Print ("Accuracy on the augmented training set: " + str(test_model(noisy_clf, train_X_noisy_testing, train_Y)))
    #Print ("Accuracy on the augmented testing set: " + str(test_model(noisy_clf, test_X_noisy_testing, test_Y)))
    #Print ("--------")
    #Print ("Randomized smoothing accuracy on the original training set: " + str(test_smooth_model (noisy_clf, train_X, train_Y, params.sigma, min_pixel_value, max_pixel_value)))
    #Print ("Randomized smoothing accuracy on the original testing set: " + str(test_smooth_model (noisy_clf, test_X, test_Y, params.sigma, min_pixel_value, max_pixel_value)))
    #Print ("Randomized smoothing accuracy on the augmented training set: " + str(test_smooth_model (noisy_clf, train_X_noisy_testing, train_Y, params.sigma, min_pixel_value, max_pixel_value)))
    #Print ("Randomized smoothing accuracy on the augmented testing set: " + str(test_smooth_model (noisy_clf, test_X_noisy_testing, test_Y, params.sigma, min_pixel_value, max_pixel_value)))
    
    Print ("")
    Print ("")
    Print ("")
    Print ("------ KERAS ------")
    
    Print ("Training the model on the original set...")
    clf = train_keras_model (train_X, train_Y, test_X, test_Y, 0, min_pixel_value, max_pixel_value)
    Print ("Training complete!")
    Print ("Training the model on the augmented set...")
    noisy_clf = train_keras_model (train_X_noisy_training, train_Y, test_X, test_Y, 0, min_pixel_value, max_pixel_value)
    Print ("Training complete!")
    
    Print ("")
    Print ("----------")
    Print ("ORIGINAL MODEL")
    Print ("----------")
    Print ("Accuracy on the original training set: " + str(test_model(clf, train_X, train_Y, isKeras=True)))
    Print ("Accuracy on the original testing set: " + str(test_model(clf, test_X, test_Y, isKeras=True)))
    Print ("Accuracy on the augmented training set: " + str(test_model(clf, train_X_noisy_testing, train_Y, isKeras=True)))
    Print ("Accuracy on the augmented testing set: " + str(test_model(clf, test_X_noisy_testing, test_Y, isKeras=True)))
    Print ("--------")
    Print ("Randomized smoothing accuracy on the original training set: " + str(test_smooth_model (clf, train_X, train_Y, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))
    Print ("Randomized smoothing accuracy on the original testing set: " + str(test_smooth_model (clf, test_X, test_Y, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))
    Print ("Randomized smoothing accuracy on the augmented training set: " + str(test_smooth_model (clf, train_X_noisy_testing, train_Y, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))
    Print ("Randomized smoothing accuracy on the augmented testing set: " + str(test_smooth_model (clf, test_X_noisy_testing, test_Y, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))
    Print ("")
    Print ("")
    Print ("----------")
    Print ("AUGMENTED MODEL")
    Print ("----------")
    Print ("Accuracy on the original training set: " + str(test_model(noisy_clf, train_X, train_Y, isKeras=True)))
    Print ("Accuracy on the original testing set: " + str(test_model(noisy_clf, test_X, test_Y, isKeras=True)))
    Print ("Accuracy on the augmented training set: " + str(test_model(noisy_clf, train_X_noisy_testing, train_Y, isKeras=True)))
    Print ("Accuracy on the augmented testing set: " + str(test_model(noisy_clf, test_X_noisy_testing, test_Y, isKeras=True)))
    Print ("--------")
    Print ("Randomized smoothing accuracy on the original training set: " + str(test_smooth_model (noisy_clf, train_X, train_Y, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))
    Print ("Randomized smoothing accuracy on the original testing set: " + str(test_smooth_model (noisy_clf, test_X, test_Y, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))
    Print ("Randomized smoothing accuracy on the augmented training set: " + str(test_smooth_model (noisy_clf, train_X_noisy_testing, train_Y, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))
    Print ("Randomized smoothing accuracy on the augmented testing set: " + str(test_smooth_model (noisy_clf, test_X_noisy_testing, test_Y, params.sigma, min_pixel_value, max_pixel_value, isKeras=True)))