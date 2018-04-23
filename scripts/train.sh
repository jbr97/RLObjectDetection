CUDA_VISIBLE_DEVICES="0" \
python trainval_net.py \
       --data_dir=data/coco/images/val2014 \
       --anno_file=data/coco/annotations/instances_minival2014.json \
       --labels_file=data/coco/labels/minival2014_action_01.json \
       --resume=/n/jbr/RL_model_dump/pretrained/faster_rcnn_new.pth \
       --batch-size=4 \
       --learning-rate=1e-2 \
       --mGPUs \
       2>&1 | tee log/train.log 
