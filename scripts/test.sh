CUDA_VISIBLE_DEVICES="0" \
python trainval_net.py -b 8 -e 4 --test \
       2>&1 | tee log/test.log 
