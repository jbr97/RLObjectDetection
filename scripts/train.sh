CUDA_VISIBLE_DEVICES="0" \
python trainval_net.py \
       --data_dir=data/coco/images/val2014 \
       --ann_file=data/coco/annotations/instances_minival2014.json \
       --dt_file=data/output/detections_minival2014_results.json \
       --pretrain=/n/jbr/RL_model_dump/pretrained/faster_rcnn_new.pth \
       --batch-size=16 \
       2>&1 | tee log/train.log 
