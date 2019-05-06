import json
import math
import os

import tensorflow as tf
from mscoco_util import *
from pycocotools.coco import COCO

data_dir = 'C:\my_files\DATA\MSCOCO'
output_path = os.path.join(data_dir, 'tfrecords')
if not os.path.exists(output_path):
    os.makedirs(output_path)
num_samples_per_file = 1000
max_bbox_per_image = 0


def json_to_tf_example(img_dir, coco_img, coco_ann):
    img_path = os.path.join(img_dir, coco_img['file_name'])
    with tf.gfile.GFile(img_path, 'rb') as file:
        encoded_jpg = file.read()
    width = coco_img['width']
    height = coco_img['height']
    filename = coco_img['file_name']
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    labels = []
    global max_bbox_per_image
    if len(coco_ann) > max_bbox_per_image:
        max_bbox_per_image = len(coco_ann)
    for ann in coco_ann:
        bbox = ann['bbox']
        xmin.append(bbox[0] / width)
        ymin.append(bbox[1] / height)
        xmax.append((bbox[0] + bbox[2]) / width)
        ymax.append((bbox[1] + bbox[3]) / height)
        labels.append(category.index(ann['category_id']))

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': bytes_feature(encoded_jpg),
                'image/filename': bytes_feature(filename.encode('utf8')),
                'image/height': int64_feature(height),
                'image/width': int64_feature(width),
                'image/bbox/xmin': float_list_feature(xmin),
                'image/bbox/ymin': float_list_feature(ymin),
                'image/bbox/xmax': float_list_feature(xmax),
                'image/bbox/ymax': float_list_feature(ymax),
                'image/bbox/label': int64_list_feature(labels),
            }))
    return example


def write_tfrecords(is_train):
    if is_train:
        record_file_name = 'coco2017_trainval_{:03d}_of_{:03d}'
        ann_file = os.path.join(data_dir,
                                'annotations/instances_train2017.json')
        img_dir = os.path.join(data_dir, 'train2017')
        coco_obj = COCO(ann_file)
    else:
        record_file_name = 'coco2017_test_{:03d}_of_{:03d}'
        ann_file = os.path.join(data_dir, 'annotations/instances_val2017.json')
        img_dir = os.path.join(data_dir, 'val2017')
        coco_obj = COCO(ann_file)

    img_ids = list(coco_obj.imgToAnns.keys())
    num_records = math.ceil(len(img_ids) / num_samples_per_file)
    for i in range(num_records):
        file_name = record_file_name.format(i, num_records)
        writer = tf.io.TFRecordWriter(os.path.join(output_path, file_name))
        ids = img_ids[i * num_samples_per_file:(i + 1) * num_samples_per_file]
        print('creating {}, with examples {}'.format(file_name, len(ids)))
        for idx, img_id in enumerate(ids):
            if idx % 100 == 0:
                print('On image %d of %d', idx, len(ids))
            coco_img = coco_obj.loadImgs(img_id)[0]
            coco_ann = coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=img_id))
            example = json_to_tf_example(img_dir, coco_img, coco_ann)
            writer.write(example.SerializeToString())
        writer.close()
    return len(img_ids)


nb_train2017 = write_tfrecords(is_train=True)
nb_val2017 = write_tfrecords(is_train=False)
print(nb_train2017)
print(nb_val2017)
print(max_bbox_per_image)
