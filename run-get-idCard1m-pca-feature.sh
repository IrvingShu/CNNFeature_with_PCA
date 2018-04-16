export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON


nohup python -u ./get_pca_feature.py \
    --image-list=../data/idcard1m/face-idcard-1M-image-list.txt \
    --feature-dir=/workspace/data/face-idcard-1M/features/insightface-r100-spa-m2.0-ep96 \
    --ipca-model-path=../model/256_ipca_split0.pkl \
    --save-format=_feat.bin \
    --out-dir=/workspace/data/face-idcard-1M/features/pca/pca-256-split0 \
    > ./logs/pca-256-split0-idCard1m-r100-spa-m2.0-ep96.txt &
#nohup python -u ./get_pca_feature.py \
#    --image-list=../data/idcard1m/face-idcard-1M-image-list.txt \
#    --feature-dir=/workspace/data/face-idcard-1M/features/insightface-r100-spa-m2.0-ep96 \
#    --ipca-model-path=../model/320_ipca_split0.pkl \
#    --save-format=_feat.bin \
#    --out-dir=/workspace/data/face-idcard-1M/features/pca/pca-320-split0 \
#    > ./logs/pca-320-split0-idCard1m-r100-spa-m2.0-ep96.txt &

#nohup python -u ./get_pca_feature.py \
#    --image-list=../data/facescrub/facescrub-mtcnn-aligned-112x112-image-list.txt \
#    --feature-dir=/workspace/data/facescrub-features/insightface-r100-spa-m2.0-ep96 \
#    --ipca-model-path=../model/320_ipca_split0.pkl \
#    --save-format=_feat.bin \
#    --out-dir=/workspace/data/facescrub-features/pca-insightface-r100-spa-m2.0-ep96/pca-320-split0 \
#    > ./logs/pca-320-split0-facescrub-r100-spa-m2.0-ep96.txt &
