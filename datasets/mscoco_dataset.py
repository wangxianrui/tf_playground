import os
import tensorflow as tf
from .preprocess.ssd_preprocessing import preprocess_image


class Param:
    IMAGE_CHN = 3
    cycle_length = 4
    nb_threads = 8
    buffer_size = 1024
    prefetch_size = 8
    data_dir = 'C:\my_files\DATA\MSCOCO'
    nb_samples_train = 117266
    nb_samples_eval = 4952
    nb_bboxs_max = 93


def parse_example_proto(example_serialized):
    """Parse the unserialized feature data from the serialized data.

    Args:
    * example_serialized: serialized example data

    Returns:
    * features: unserialized feature data
    """

    # parse features from the serialized data
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], \
            dtype=tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature([], \
            dtype=tf.string, default_value=''),
        'image/height': tf.FixedLenFeature([], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([], dtype=tf.int64),
        'image/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }
    features = tf.parse_single_example(example_serialized, feature_map)

    return features


def pack_annotations(bboxes, labels, difficults=None, truncateds=None):
    """Pack all the annotations into one tensor.

    Args:
    * bboxes: list of bounding box coordinates (N x 4)
    * labels: list of category labels (N)
    * difficults: list of difficulty flags (N)
    * truncateds: list of truncation flags (N)

    Returns:
    * objects: one tensor with all the annotations packed together (FLAGS.nb_bboxs_max x 8)
    """

    # pack <bboxes> & <labels> with a leading <flags>
    labels = tf.cast(tf.expand_dims(labels, 1), tf.float32)
    flags = tf.ones(tf.shape(labels))
    objects = tf.concat([flags, bboxes, labels], axis=1)

    # pad to fixed number of bounding boxes
    pad_size = Param.nb_bboxs_max - tf.shape(objects)[0]
    objects = tf.pad(objects, [[0, pad_size], [0, 0]])

    return objects


def parse_fn(example_serialized, is_train, img_size):
    """Parse image & objects from the serialized data.

    Args:
    * example_serialized: serialized example data
    * is_train: whether to construct the training subset

    Returns:
    * image: image tensor
    * objects: one tensor with all the annotations packed together
    """

    # unserialize the example proto
    features = parse_example_proto(example_serialized)

    # obtain the image data
    image_raw = tf.image.decode_jpeg(
        features['image/encoded'], channels=Param.IMAGE_CHN)
    filename = features['image/filename']
    shape = tf.convert_to_tensor(
        [features['image/height'], features['image/width'], 3])
    xmins = tf.expand_dims(features['image/bbox/xmin'].values, 1)
    ymins = tf.expand_dims(features['image/bbox/ymin'].values, 1)
    xmaxs = tf.expand_dims(features['image/bbox/xmax'].values, 1)
    ymaxs = tf.expand_dims(features['image/bbox/ymax'].values, 1)
    bboxes_raw = tf.concat([ymins, xmins, ymaxs, xmaxs], axis=1)
    labels_raw = tf.cast(features['image/bbox/label'].values, tf.int64)

    # pre-process image, labels, and bboxes
    if isinstance(img_size, int):
        img_size = [img_size, img_size]
    if is_train:
        image, labels, bboxes = preprocess_image(
            image_raw,
            labels_raw,
            bboxes_raw,
            img_size,
            is_training=True,
            data_format='channels_last',
            output_rgb=False)
    else:
        image = preprocess_image(
            image_raw,
            labels_raw,
            bboxes_raw,
            img_size,
            is_training=False,
            data_format='channels_last',
            output_rgb=False)
        labels, bboxes = labels_raw, bboxes_raw

    # pack all the annotations into one tensor
    img_info = {'filename': filename, 'shape': shape}
    groundtruth = pack_annotations(bboxes, labels)
    return img_info, image, groundtruth


class MSCOCODataset:
    """MSCOCODataset."""

    def __init__(self, is_train, img_size):
        assert Param.data_dir is not None, '<data_dir> must not be None'
        if is_train:
            self.file_pattern = os.path.join(Param.data_dir, '*trainval*of*')
        else:
            self.file_pattern = os.path.join(Param.data_dir, '*test*of*')
        self.dataset_fn = lambda x: tf.data.TFRecordDataset(x)
        self.parse_fn = lambda x: parse_fn(x, is_train, img_size)

    def build(self, batch_size):
        filenames = tf.data.Dataset.list_files(self.file_pattern)
        dataset = filenames.apply(
            tf.data.experimental.parallel_interleave(
                self.dataset_fn, cycle_length=Param.cycle_length))
        dataset = dataset.map(
            self.parse_fn, num_parallel_calls=Param.nb_threads)
        dataset = dataset.shuffle(
            Param.buffer_size).batch(batch_size).repeat().prefetch(
                Param.prefetch_size)
        return dataset.make_one_shot_iterator()

    def check_count(self):
        nums = 0
        filenames = tf.gfile.Glob(
            os.path.join(Param.data_dir, '*trainval*of*'))
        for name in filenames:
            for string_record in tf.io.tf_record_iterator(name):
                nums += 1
        print(nums)
        nums = 0
        filenames = tf.gfile.Glob(os.path.join(Param.data_dir, '*test*of*'))
        for name in filenames:
            for string_record in tf.io.tf_record_iterator(name):
                nums += 1
        print(nums)
