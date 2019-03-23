import tensorflow as tf
import os
from .preprocess.imagenet_preprocessing import preprocess_image


class Param:
    IMAGE_HEI = 224
    IMAGE_WID = 224
    IMAGE_CHN = 3
    cycle_length = 4
    nb_threads = 8
    buffer_size = 1024
    prefetch_size = 8
    data_dir = ''
    nb_classes = 1000
    nb_samples_train = None
    nb_samples_eval = None


def parse_example_proto(example_serialized):
    """Parse image buffer, label, and bounding box from the serialized data.

    Args:
    * example_serialized: serialized example data

    Returns:
    * image_buffer: image buffer label
    * label: label tensor (not one-hot)
    * bbox: bounding box tensor
    """

    # parse features from the serialized data
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    bbox_keys = ['image/object/bbox/' + x for x in ['xmin', 'ymin', 'xmax', 'ymax']]
    feature_map.update({key: tf.VarLenFeature(dtype=tf.float32) for key in bbox_keys})
    features = tf.parse_single_example(example_serialized, feature_map)

    # obtain the label and bounding boxes
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox


def parse_fn(example_serialized, is_train):
    """Parse image & labels from the serialized data.

    Args:
    * example_serialized: serialized example data
    * is_train: whether data augmentation should be applied

    Returns:
    * image: image tensor
    * label: one-hot label tensor
    """

    image_buffer, label, bbox = parse_example_proto(example_serialized)
    image = preprocess_image(
        image_buffer=image_buffer, bbox=bbox, output_height=Param.IMAGE_HEI,
        output_width=Param.IMAGE_WID, num_channels=Param.IMAGE_CHN, is_training=is_train)
    label = tf.one_hot(tf.reshape(label, []), Param.nb_classes)

    return image, label


class Ilsvrc12Dataset:
    """
    ILSVRC-12 dataset
        file_pattern
        dataset_fn
        parse_fn
    """

    def __init__(self, is_train):
        assert Param.data_dir is not None, '<data_dir> must not be None'
        if is_train:
            self.file_pattern = os.path.join(Param.data_dir, 'train-*-of-*')
        else:
            self.file_pattern = os.path.join(Param.data_dir, 'validation-*-of-*')
        self.dataset_fn = lambda x: tf.data.TFRecordDataset(x)
        self.parse_fn = lambda x: parse_fn(x, is_train=is_train)

    def build(self, batch_size):
        filenames = tf.data.Dataset.list_files(self.file_pattern)
        dataset = filenames.apply(
            tf.data.experimental.parallel_interleave(self.dataset_fn, cycle_length=Param.cycle_length))
        dataset = dataset.map(self.parse_fn, num_parallel_calls=Param.nb_threads)
        dataset = dataset.shuffle(Param.buffer_size).batch(batch_size).repeat().prefetch(Param.prefetch_size)
        return dataset.make_one_shot_iterator()
