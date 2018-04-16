import os
import sys
import json
import struct
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib


cv_type_to_dtype = {
        5 : np.dtype('float32'),
        6 : np.dtype('float64')
}

def read_mat(f):
    """
    Reads an OpenCV mat from the given file opened in binary mode
    """
    rows, cols, stride, type_ = struct.unpack('iiii', f.read(4*4))
    mat = np.fromstring(f.read(rows*stride),dtype=cv_type_to_dtype[type_])
    return mat.reshape(rows,cols)

if __name__ == "__main__":
    feature_dir = '/workspace/data/ms1m-features/insightface-r100-spa-m2.0-ep96'
    #feature_list_path = '../data/ms1m/ms1m_insightface_112x112_rgb.txt'
    feature_list_path = '../data/ms1m/ms1m_insightface_112x112_rgb_split_0.txt'
    save_type = '_feat.bin'
    
    #X = np.zeros(shape=(475605, 512))
    i = 0
    with open(feature_list_path, 'r') as f:
        lines = f.readlines()
        print('###### read features nums: %d ######' %(len(lines)))
        X= np.zeros(shape=(len(lines), 512))       
 
        for line in lines:
            feature_name = line.strip() + save_type
            with open(os.path.join(feature_dir, feature_name)) as f1:
                x_vec = np.transpose(read_mat(f1))
                X[i] = x_vec
                i = i + 1
    print('###### success load feature nums: %d ######'%i)
    print(X.shape)
    #ipca   
    ipca = IncrementalPCA(n_components=320)
    ipca.fit(X)
    print('###### PCA Done! ######')
    joblib.dump(ipca, '../model/320_ipca_all.pkl')
    print('components num: %d' %ipca.n_components)
    #print('explained_variance_ratio: %d' % ipca.explained_variance_ratio_)
    sum_variance_ratio = 0
    for i in range(ipca.n_components):
        sum_variance_ratio += ipca.explained_variance_ratio_[i]
    print('sum_variance_ratio: %f' %sum_variance_ratio) 
