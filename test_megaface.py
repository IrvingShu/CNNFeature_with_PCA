import os
import sys
import os.path as osp
import json
import struct
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib
import matio


if __name__ == "__main__":
    feature_dir = '/workspace/data/megaface-features/insightface-r100-spa-m2.0-ep96'
    feature_list_path = os.path.join('../data/megaface/megaface-image-list-aligned-112x112.txt')
    
    #load pca model
    ipca = joblib.load('../model/320_ipca_split0.pkl')
    #ipca = joblib.load('../model/384_ipca_all.pkl')
    print('components num: %d' %ipca.n_components)
    print('explained_variance_ratio: %s' % str(ipca.explained_variance_ratio_.shape))
    sum_variance_ratio = 0
    for i in range(ipca.n_components):
        sum_variance_ratio += ipca.explained_variance_ratio_[i]
    print('sum_variance_ratio: %f' %sum_variance_ratio)

    save_type = '_feat.bin'
    save_dir = '/workspace/data/megaface-features/pca-insightface-r100-spa-m2.0-ep96/pca-320-split0'
    
    if not osp.exists(save_dir):
        os.makedirs(save_dir) 
    
    succ_num = 0 
    failed_num = 0
    with open(feature_list_path, 'r') as f:
        lines = f.readlines()
        print('###### read features nums: %d ######' %(len(lines)))        
        for line in lines:
            sub_dir_list = line.split('/')
            sub_dir = ''
            for i in range(len(sub_dir_list) - 1):
                sub_dir = osp.join(sub_dir, sub_dir_list[i])
            final_dir = os.path.join(save_dir, sub_dir)
            
            if not osp.exists(final_dir):
                os.makedirs(final_dir)

            feature_path = osp.join(feature_dir, line.strip()+save_type)
            new_feature_path = osp.join(final_dir, line.strip().split('/')[-1] + save_type)
            if not osp.exists(feature_path):
                print('Not existed: %s' %feature_path)
                failed_num = failed_num + 1
            else:
                x_vec = np.transpose(matio.load_mat(feature_path))
                new_x_vec = ipca.transform(x_vec)
                matio.save_mat(new_feature_path, new_x_vec.T)
                succ_num = succ_num + 1
                if succ_num % 100000 == 0:
                    print('####### Save index %d feature  ######' %succ_num)
    print('###### Finished Feature Reduce Dims, %d ######' %succ_num)
    print('###### Failed Feature Reduce Dims, %d ######' %failed_num)
    
