export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON

nohup python -u ./train_ipca.py \
    --image-list=../data/ms1m/ms1m_insightface_112x112_rgb_split_0.txt \
    --feature-dir=/workspace/data/ms1m-features/insightface-r100-spa-m2.0-ep96 \
    --feature-dims=512 \
    --save-format=_feat.bin \
    --ipca-save-path=../model/360_ipca_all.pkl \
    --n_components=360 \
    > ./logs/train-360-split0-fgnet-r100-spa-m2.0-ep96.txt &
