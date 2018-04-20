import sys
sys.path.append('/home/zhenglf/liyujun/RL_on_ObjectDetection/lib')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

import json
import numpy as np

gtpath = '/S2/MI/data/human_pose/mscoco/annotations/instances_minival2014.json'
pdpath = '/S2/MI/jbr/RLObjectDetection/output/res101/coco_2014_minival/faster_rcnn_10_validation/detections_minival2014_results.json'

# f = open(gtpath, 'r')
# a = json.load(f)
# f.close()

# for i in range(10):
#     print(a['annotations'][i][u'image_id'])

# print('gt len:', len(a['annotations']))

f2 = open(pdpath, 'r')
b = json.load(f2)
f2.close()

imageidlist = [x['image_id'] for x in b]
imageidlist = sorted(np.unique(imageidlist))
print('len imageidlist:', len(imageidlist))
print('imageidlist[0]:', imageidlist[0])
print('---------------------------')

imgID2annID = {}

for counter, x in enumerate(b):
    image_id = x['image_id']
    if image_id not in imgID2annID:
        imgID2annID[image_id] = [counter]
    else:
        imgID2annID[image_id].append(counter)

print('len imgid2annid:', len(imgID2annID))
print('imgid2annid[139]:', imgID2annID[139])
print('-----------------------')

counter = 1
for image_id in imageidlist:
    DtIds = imgID2annID[image_id]
    for dtid in DtIds:
        dtann = b[dtid]
        print('dtann:', dtann)
       
    if counter < 5:
        break 

print('------------------------')
cocoGt = COCO(gtpath)
cocoDt = COCO(pdpath)
imgIDs = sorted(cocoGt.getImgIds())
catIDs = sorted(cocoGt.getCatIds())

gts = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=imgIDs, catIds=catIDs))
dts = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=imgIDs))
print('len gts:', len(gts))
print('len dts:', len(dts))
print('-------------------------')
print(gts[0])

imgIDsDt = sorted(cocoDt.getImgIds())
print('len imgIDs:', len(imgIDs))
print('len imgIDsDt:', len(imgIDsDt))
print('-------------------------')

gt = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[139]))
print('gt[0]:', gt[0])
print('------------------------')

