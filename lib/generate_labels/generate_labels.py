import time
import sys
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

import json
import numpy as np

# Percentage movement on (x,y,w,h).
# Each movement will generate a json file, consisting of 'dious', 'act', 'image_id', 'bbox', 'score', 'category_id'.
acts = [[-0.02, 0, 0, 0], [0, -0.02, 0, 0], [0, 0, -0.02, 0], [0, 0, 0, -0.02]]

# gtpah is groudtruth path. pdpath is the result path.
# gtpath = '/S2/MI/data/human_pose/mscoco/annotations/instances_minival2014.json'
# pdpath = '/S2/MI/jbr/RLObjectDetection/output/res101/coco_2014_minival/faster_rcnn_10_validation/detections_minival2014_results.json'
gtpath = '/S2/MI/data/human_pose/mscoco/annotations/instances_train2014.json'
pdpath = '/S2/MI/jbr/RLObjectDetection/output/res101/coco_2014_train/faster_rcnn_10/detections_train2014_results.json'
jsonname = 'train2014'


f = open(pdpath, 'r')
dts = json.load(f)
f.close()

imageidlist = [x['image_id'] for x in dts]
imageidlist = sorted(np.unique(imageidlist))
print('---------------------------')
print('len imageidlist:', len(imageidlist))
print('imageidlist[0]:', imageidlist[0])
print('---------------------------')


_dts = defaultdict(list)
for dt in dts:
    _dts[dt['image_id'], dt['category_id']].append(dt)


print('------------------------')
cocoGt = COCO(gtpath)
imgIDs = sorted(cocoGt.getImgIds())
catIDs = sorted(cocoGt.getCatIds())

print('len imgIDs:', len(imgIDs))

gts = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=imgIDs, catIds=catIDs))
print('len gts:', len(gts))

_gts = defaultdict(list)
for gt in gts:
    _gts[gt['image_id'], gt['category_id']].append(gt)

print('-------------------------')

def computeIoU(imgId, catId, useCats=False, maxDets=100000):
    if useCats:
        gt = _gts[imgId, catId]
        dt = _gts[imgId, catId]
    else:
        gt = [_ for cId in catIDs for _ in _gts[imgId, cId]]
        dt = [_ for cId in catIDs for _ in _dts[imgId, cId]]

    if len(gt) == 0 or len(dt) == 0:
        return [], 1, 1, 1

    dt = sorted(dt, key=lambda x: -x['score'])
    if len(dt) > maxDets:
        dt = dt[0:maxDets]

    g = [g['bbox'] for g in gt]
    d = [d['bbox'] for d in dt]

    # print('d len:', len(d))
    # print('g len:', len(g))

    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = COCOmask.iou(d, g, iscrowd)
    ious = np.array(ious)

    if ious.ndim == 0:
        ious = np.array([ious])

    if ious.ndim == 1:
        if len(g) > 1:
            ious = np.array([ max(ious) ])

    if ious.ndim == 2:
        ious = np.amax(ious, axis=1)

    if ious.ndim > 2:
        print('Wrong in ious dim!')
        sys.exit(-1)

    return ious, g, dt, iscrowd

# action is 4-tuple floats, [0.02, 0, 0, 0]
def computeNewIoU(g, d, iscrowd, action):
    if g == 1:
        return []

    for i in range(len(d)):
        x, y, w, h = d[i]
        dx = w * action[0]
        dy = h * action[1]
        dw = w * action[2]
        dh = h * action[3]
        
        x += dx
        y += dy
        w += dw
        h += dh
        d[i] = [x, y, w, h]

    ious = COCOmask.iou(d, g, iscrowd)
    ious = np.array(ious)

    if ious.ndim == 0:
        ious = np.array([ious])
        return ious
    if ious.ndim == 1:
        if len(d) > 1:
            return ious
        else:
            ious = np.array([ max(ious) ])
            return ious
    if ious.ndim == 2:
        ious = np.amax(ious, axis=1)
        return ious

    if ious.ndim > 2:
        print('Wrong in newious dim!')
        sys.exit(-1)

# ious, g, dt, iscrowd = computeIoU(139, 86)
# # print('iou sample:', ious)
# 
# d = [d['bbox'] for d in dt]
# act = [0.02, 0, 0, 0]
# newious = computeNewIoU(g, d, iscrowd, act)
# # print('newious:', newious)
# 
# for i in range(len(dt)):
#     dious = newious[i] - ious[i]
#     dt[i]['dious'] = dious
#     dt[i]['act'] = act


# f = open('1.json', 'w')
# json.dump(dt, f)
# f.close() 

   
for act in acts:
    tic = time.time()
    dtlist = []
    cnt = -1 
    for image_id in imgIDs:
        print('image_id :', image_id)
        sys.exit("exit suddenly.")
        cnt += 1
        if cnt % 10000 == 0:
            print('num: {:8d} | image_id:{:10d}'.format(cnt, image_id))
        # for category_id in catIDs:
        ious, g, dt, iscrowd = computeIoU(image_id, 0)
        if g == 1:
            continue

        d = [d['bbox'] for d in dt] 
        newious = computeNewIoU(g, d, iscrowd, act)

        if len(dt) == 0:
            print('ious:', ious)    

        if len(ious) == 0:
            print('ious::', ious, 'g::', g)
            print('len(ious):', len(ious))

        for i in range(len(dt)):
            dious = newious[i] - ious[i]
            dt[i]['dious'] = dious
            dt[i]['act'] = act 
    
        dtlist += dt

        # break

    f = open('{}{}.json'.format(jsonname, act), 'w')
    json.dump(dtlist, f)
    f.close()
    toc = time.time()
    print('--- time: {:.4f} seconds ---'.format(toc - tic))
