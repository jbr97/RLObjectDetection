CUDA_VISIBLE_DEVICES="3" \
python trainval_net.py \
       --data_dir=/home/jbr/storage-jbr/RLObjectDetection/data/coco/images/train2014 \
       --anno_file=/home/jbr/storage-jbr/RLObjectDetection/data/coco/annotations/instances_train2014.json \
       --labels_file=/home/jbr/storage-jbr/RLObjectDetection/data/coco/labels/train2014_action_01.json \
       --resume=snapshot/pretrained/faster_rcnn_new.pth \
       --batch-size=16 \
       --learning-rate=1e-2 \
       2>&1 | tee log/train.log 
