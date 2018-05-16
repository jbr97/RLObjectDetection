CUDA_VISIBLE_DEVICES="0" \
python trainval_net.py -b 1 -e 1 --test \
       2>&1 | tee log/test.log 
