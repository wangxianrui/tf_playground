import json
import math
import os

import tensorflow as tf
from mscoco_util import *
from pycocotools.coco import COCO

data_dir = 'D:/dataset/mscoco_2017'
output_path = os.path.join(data_dir, 'tfrecords')
num_samples_per_file = 1000


def json_to_tf_example(img_dir, coco_img, coco_ann):
    img_path = os.path.join(img_dir, coco_img['file_name'])
    with tf.gfile.GFile(img_path, 'rb') as file:
        encoded_jpg = file.read()
    width = coco_img['width']
    height = coco_img['height']
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    print(coco_ann)
    input()

    return 0


def write_tfrecords(is_train):
    if is_train:
        record_file_name = 'coco2017_trainval_{:03d}_of_{:03d}'
        ann_file = os.path.join(data_dir, 'annotations/instances_train2017.json')
        img_dir = os.path.join(data_dir, 'images/train2017')
        coco_obj = COCO(ann_file)
    else:
        record_file_name = 'coco2017_test_{:03d}_of_{:03d}'
        ann_file = os.path.join(data_dir, 'annotations/instances_val2017.json')
        img_dir = os.path.join(data_dir, 'images/val2017')
        coco_obj = COCO(ann_file)
    ##
    # imgids = coco_obj.getImgIds()
    # print(len(imgids))
    ##

    img_ids = coco_obj.imgToAnns.keys()
    num_records = math.ceil(len(img_ids) / num_samples_per_file)
    for i in range(num_records):
        file_name = record_file_name.format(i, num_records)
        writer = tf.io.TFRecordWriter(os.path.join(output_path, file_name))
        ids = img_ids[i * num_samples_per_file: (i + 1) * num_samples_per_file]
        tf.logging.info('creating {}, with examples {}'.format(file_name, len(ids)))
        for idx, img_id in enumerate(ids):
            if idx % 100 == 0:
                tf.logging.info('On image %d of %d', idx, len(ids))
            coco_img = coco_obj.loadImgs(img_id)
            coco_ann = coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=img_id))
            example = json_to_tf_example(img_dir, coco_img, coco_ann)
            writer.write(example.SerializeToString())
        writer.close()


write_tfrecords(is_train=True)
write_tfrecords(is_train=False)
