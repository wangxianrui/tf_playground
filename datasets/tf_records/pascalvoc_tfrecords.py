import math
import os
import xml.etree.ElementTree as ET

from pascalvoc_util import *

data_dir = 'C:\my_files\DATA\VOCdevkit'
output_path = os.path.join(data_dir, 'tfrecords')
if not os.path.exists(output_path):
    os.makedirs(output_path)
ignore_difficult_instances = False
num_samples_per_file = 1000
max_bbox_per_image = 0


def dict_to_tf_example(xml_data):
    img_path = os.path.join(data_dir, xml_data['folder'], 'JPEGImages',
                            xml_data['filename'])
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    width = int(xml_data['size']['width'])
    height = int(xml_data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    labels = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in xml_data:
        global max_bbox_per_image
        if len(xml_data['object']) > max_bbox_per_image:
            max_bbox_per_image = len(xml_data['object'])
        for obj in xml_data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue
            difficult_obj.append(int(difficult))
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            labels.append(label_map[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': bytes_feature(encoded_jpg),
                'image/filename': \
                    bytes_feature(xml_data['filename'].encode('utf8')),
                'image/height': int64_feature(height),
                'image/width': int64_feature(width),
                'image/bbox/xmin': float_list_feature(xmin),
                'image/bbox/xmax': float_list_feature(xmax),
                'image/bbox/ymin': float_list_feature(ymin),
                'image/bbox/ymax': float_list_feature(ymax),
                'image/bbox/label': int64_list_feature(labels),
                'image/bbox/difficult': int64_list_feature(difficult_obj),
                'image/bbox/truncated': int64_list_feature(truncated),
            }))
    return example


def write_tfrecords(is_training, year):
    if is_training:
        record_file_name = '{}_trainval_{:03d}_of_{:03d}'
        image_list_path = os.path.join(data_dir, year, \
            'ImageSets', 'Main', 'aeroplane_' + 'trainval' + '.txt')

    else:
        record_file_name = '{}_test_{:03d}_of_{:03d}'
        image_list_path = os.path.join(data_dir, year, \
            'ImageSets', 'Main', 'aeroplane_' + 'test' + '.txt')
    annotations_dir = os.path.join(data_dir, year, 'Annotations')
    image_list = read_examples_list(image_list_path)
    record_nums = math.ceil(len(image_list) / num_samples_per_file)

    for i in range(int(record_nums)):
        file_name = record_file_name.format(year, i, record_nums)
        writer = tf.io.TFRecordWriter(os.path.join(output_path, file_name))
        images = image_list[i * num_samples_per_file:(i + 1) *
                            num_samples_per_file]
        print('creating {}, with examples {}'.format(file_name, len(images)))
        for idx, image in enumerate(images):
            if idx % 100 == 0:
                print('On image %d of %d', idx, len(images))
            path = os.path.join(annotations_dir, image + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = ET.fromstring(xml_str)
            xml_data = recursive_parse_xml_to_dict(xml)['annotation']
            example = dict_to_tf_example(xml_data)
            writer.write(example.SerializeToString())
        writer.close()
    return len(image_list)


nb_test_2007 = write_tfrecords(False, 'VOC2007')
nb_traival_2007 = write_tfrecords(True, 'VOC2007')
nb_trainval_2012 = write_tfrecords(True, 'VOC2012')
print(nb_test_2007)
print(nb_traival_2007)
print(nb_trainval_2012)
print(max_bbox_per_image)
