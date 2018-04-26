CUDA_VISIBLE_DEVICES="0" \
python trainval_net.py \
       --data_dir=data/coco/images/train2014 \
       --ann_file=data/coco/annotations/instances_train2014.json \
       --dt_file=data/output/detections_train2014_results.json \
       --pretrain=/n/jbr/RL_model_dump/pretrained/faster_rcnn_new.pth \
       --batch-size=16 \
       2>&1 | tee log/train.log 
