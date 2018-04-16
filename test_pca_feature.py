import os
import sys
import os.path as osp
import json
import struct
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib
from sklearn import preprocessing
import matio
import argparse

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-list', type=str, help='image list file')
    parser.add_argument('--feature-dir', type=str, help='feature dir')
    parser.add_argument('--ipca-model-path', type=str, help='ipca model path')
    parser.add_argument('--save-format', type=str, help='feature format')
    parser.add_argument('--out-dir', type=str, help='where to save the feature after pca')
    return parser.parse_args(argv)

def main(args):
    print('===> args:\n', args)
    
    image_list = args.image_list
    feature_dir = args.feature_dir
    
    ipca_model_path = args.ipca_model_path   
    save_format = args.save_format
     
    out_dir = args.out_dir
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    #load pca model
    ipca = joblib.load(ipca_model_path)
    
    print('components num: %d' %ipca.n_components)
    print('explained_variance_ratio: %s' % str(ipca.explained_variance_ratio_.shape))
    sum_variance_ratio = 0
    for i in range(ipca.n_components):
        sum_variance_ratio += ipca.explained_variance_ratio_[i]
    print('sum_variance_ratio: %f' %sum_variance_ratio)

    succ_num = 0 
    failed_num = 0
    with open(image_list, 'r') as f:
        lines = f.readlines()
        print('###### read features nums: %d ######' %(len(lines)))        
        for line in lines:
            sub_dir_list = line.split('/')
            sub_dir = ''
            for i in range(len(sub_dir_list) - 1):
                sub_dir = osp.join(sub_dir, sub_dir_list[i])
            final_dir = os.path.join(out_dir, sub_dir)
            
            if not osp.exists(final_dir):
                os.makedirs(final_dir)
            feature_name = line.strip() + save_format
            feature_path = osp.join(feature_dir, feature_name)
            new_feature_path = osp.join(final_dir, line.strip().split('/')[-1] + save_format)
            if not osp.exists(feature_path):
                print('Not existed: %s' %feature_path)
                failed_num = failed_num + 1
            else:
                x_vec = np.transpose(matio.load_mat(feature_path))
                #print('pca after')
                x_vec_by_pca = ipca.transform(x_vec)
             
                feature_len = x_vec_by_pca.shape[1]
                norm_x_vec = np.zeros(shape=(1,feature_len))
                 
                norm_x_vec = 1.0 * x_vec_by_pca /(np.linalg.norm(x_vec_by_pca, ord=2) + 0.000001)               
                #feature is NAN
                if np.isnan(norm_x_vec).sum() > 0:
                    print('%s is Nan' %feature_path) 
                
                else: 
                    matio.save_mat(new_feature_path, norm_x_vec.T)
                    succ_num = succ_num + 1
                    if succ_num % 10000 == 0:
                        print('####### Save index %d feature  ######' %succ_num)
    print('###### Finished Feature Reduce Dims, %d ######' %succ_num)
    print('###### Failed Feature Reduce Dims, %d ######' %failed_num)
    
if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
