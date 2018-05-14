CUDA_VISIBLE_DEVICES="0" \
python trainval_net.py -b 32 \
       2>&1 | tee log/train.log 
