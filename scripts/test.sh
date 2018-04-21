CUDA_VISIBLE_DEVICES="3" \
  python trainval_net.py \
       --data_dir=data/coco/images/val2014 \
       --anno_file=data/coco/annotations/instances_minival2014.json \
       --labels_file=data/coco/labels/minival2014_action_01.json \
       --resume=snapshot/epoch_1.pth \
       --batch-size=1 \
       --evaluate \
       2>&1 | tee log/test.log 
