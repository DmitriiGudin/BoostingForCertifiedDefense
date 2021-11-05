import numpy as np
import time
import pickle
import h5py
import params


def classify_smoothed_image (image, pixel_max_value, model, N_MC, gauss): # 
    classes = []
    for n in range(N_MC):
        noised_image = image+np.random.normal(0, np.sqrt(gauss)*pixel_max_value, len(image))
        noised_image = np.array([max(p,0) for p in noised_image])
        noised_image = np.array([min(p,pixel_max_value) for p in noised_image])
        classes.append(model.predict([noised_image])[0])
    return np.array(classes)
    
    
if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)
        
    with open(params.model_filename, 'rb') as f:
        clf = pickle.load(f)

    hdf5_read_file = h5py.File(params.training_file,'r')
    hdf5_write_file = h5py.File(params.smoothed_training_file,'w')
    X, Y = hdf5_read_file['X'][:], hdf5_read_file['Y'][:]
    dset = hdf5_write_file.create_dataset('Y', (len(Y),params.N_MC), int)
    pixel_max_value = max(Y)
    for i, im in enumerate(X):
        dset[i] = classify_smoothed_image (im, pixel_max_value, clf, params.N_MC, params.gauss)
        Print ('Image ' + str(i+1) + ' out of ' + str(len(Y)) + ' done.')
        
    hdf5_read_file.close()
    hdf5_write_file.close()