export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON

nohup python -u ./get_pca_feature.py \
    --image-list=../data/fgnet/fgnet-image-list-aligned-112x112.txt \
    --feature-dir=/workspace/data/fgnet-features/insightface-r100-spa-m2.0-ep96/ \
    --ipca-model-path=../model/320_ipca_split0.pkl \
    --save-format=_feat.bin \
    --out-dir=/workspace/data/fgnet-features/pca-insightface-r100-spa-m2.0-ep96/pca-320-split0 \
    > ./logs/test2_pca-320-split0-fgnet-r100-spa-m2.0-ep96.txt &
