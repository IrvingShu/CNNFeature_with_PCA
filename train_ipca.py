import os
import sys
import json
import struct
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib

import matio
import argparse

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-list', type=str, help='image list file')
    parser.add_argument('--feature-dir', type=str, help='feature dir') 
    parser.add_argument('--feature-dims', type=int, help='feature dims', default=512) 
    parser.add_argument('--save-format', type=str, help='feature format')
    parser.add_argument('--ipca-save-path', type=str, help='ipca model save path', default= '../model/pca_model.pkl')
    parser.add_argument('--n_components', type=int, help='PCA n_components', default=2) 
    return parser.parse_args(argv)

def main(args):
    print('===> args:\n', args)

    image_list = args.image_list
    feature_dir = args.feature_dir
    
    save_type = args.save_format
    feature_len = args.feature_dims

    i = 0
    with open(image_list, 'r') as f:
        lines = f.readlines()
        print('###### read features nums: %d ######' %(len(lines)))
        X= np.zeros(shape=(len(lines), feature_len))       
 
        for line in lines:
            feature_name = line.strip() + save_type
            feature_path = os.path.join(feature_dir, feature_name)
            x_vec = np.transpose(matio.load_mat(feature_path))
            X[i] = x_vec
            i = i + 1
    print('###### success load feature nums: %d ######'%i)
    print(X.shape)
    #ipca   
    ipca = IncrementalPCA(n_components=args.n_components)
    ipca.fit(X)
    print('###### PCA Done! ######')
    joblib.dump(ipca, args.ipca_save_path)

    print('components num: %d' %ipca.n_components)
    sum_variance_ratio = 0
    for i in range(ipca.n_components):
        sum_variance_ratio += ipca.explained_variance_ratio_[i]
    print('sum_variance_ratio: %f' %sum_variance_ratio)

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))

 
