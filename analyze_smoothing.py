import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import pickle
from statistics import mode
import params


smoothed_file = params.smoothed_training_file
datafile = params.training_file


def get_confusion_matrix (Y, Y_true):
    confusion_matrix = np.zeros((len(params.classes),len(params.classes)))
    for i in range(len(params.classes)):
        for j in range(len(params.classes)):
            confusion_matrix[i,j] = len([0 for k in range(len(Y)) if Y_true[k]==i and Y[k]==j])
    return confusion_matrix
    
    
def plot_confusion_matrix (Y, Y_true):
    confusion_matrix = get_confusion_matrix(Y, Y_true)
    confusion_matrix = confusion_matrix/len(Y)
    df = pd.DataFrame(confusion_matrix, columns=params.classes, index=params.classes)
    plt.figure(figsize=(len(params.classes),len(params.classes)))
    heatmap = sns.heatmap(df, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Confusion Matrix', fontdict={'fontsize':18}, pad=12)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/confusion_matrix.png", dpi=100)
    plt.gcf().savefig("plots/confusion_matrix.eps", dpi=100)
    plt.close()
    
    
def plot_hist_smoothing_distributions (Y_true, Y_smoothed, pred_type=True): # If pred_type is True, the frequency is computed for the true label; if False, then for the non-smoothed model-predicted label.
    array = np.array([len([0 for img_Y in Y_smoothed[i] if img_Y==Y_true[i]]) for i in range(len(Y_true))])
    plt.clf()
    plt.title("Randomized smoothing", size=24)
    if pred_type:
        plt.xlabel('True prediction frequency', size=24)
    else:
        plt.xlabel('Initial prediction frequency', size=24)
    plt.ylabel('Occurrences', size=24)
    plt.tick_params(labelsize=18)
    plt.hist(array, bins=50, color='black',fill=False,linewidth=1,histtype='step')
    plt.gcf().set_size_inches(25.6, 14.4)
    if pred_type:
        plt.gcf().savefig("plots/hist_smoothing_distributions_true.png", dpi=100)
        plt.gcf().savefig("plots/hist_smoothing_distributions_true.eps", dpi=100)
    else:
        plt.gcf().savefig("plots/hist_smoothing_distributions_pred.png", dpi=100)
        plt.gcf().savefig("plots/hist_smoothing_distributions_pred.eps", dpi=100)
    plt.close()


if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)

    with open(params.model_filename, 'rb') as f:
        clf = pickle.load(f)

    hdf5_file = h5py.File(smoothed_file, 'r')
    hdf5_datafile = h5py.File(datafile, 'r')
    Y_smoothed = hdf5_file['Y'][:]
    X_true = hdf5_datafile['X'][:]
    Y_true = hdf5_datafile['Y'][:]
    Y_frequent = np.array([mode(c) for c in Y_smoothed])
    Y_pred = np.array([clf.predict([image])[0] for image in X_true])
    
    Print ("Accuracy: " + str(len([0 for k in range(len(Y_true)) if Y_frequent[k]==Y_true[k]])/len(Y_true)))
    Print ("Confusion matrix: " + str(get_confusion_matrix(Y_frequent, Y_true)))
    
    plot_hist_smoothing_distributions (Y_true, Y_smoothed, pred_type=True)
    plot_hist_smoothing_distributions (Y_pred, Y_smoothed, pred_type=False)
    plot_confusion_matrix (Y_frequent, Y_true)