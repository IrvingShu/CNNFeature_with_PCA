export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON

nohup python -u ./get_pca_feature.py \
    --image-list=../data/megaface/megaface-image-list-aligned-112x112.txt \
    --feature-dir=/workspace/data/megaface-features/insightface-r100-spa-m2.0-ep96 \
    --ipca-model-path=../model/384_ipca_all.pkl \
    --save-format=_feat.bin \
    --out-dir=/workspace/data/megaface-features/pca-insightface-r100-spa-m2.0-ep96/pca-384-all \
    > ./logs/pca-384-all-megaface-r100-spa-m2.0-ep96.txt &
