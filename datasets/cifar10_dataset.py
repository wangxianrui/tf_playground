import os

import tensorflow as tf


class Param:
    LABEL_BYTES = 1
    IMAGE_HEI = 32
    IMAGE_WID = 32
    IMAGE_CHN = 3
    IMAGE_BYTES = IMAGE_CHN * IMAGE_HEI * IMAGE_WID
    RECORD_BYTES = LABEL_BYTES + IMAGE_BYTES
    IMAGE_AVE = [125.3, 123.0, 113.9]
    IMAGE_STD = [63.0, 62.1, 66.7]
    cycle_length = 4
    nb_threads = 8
    buffer_size = 1024
    prefetch_size = 8
    data_dir = 'D:\dataset\cifar-10-binary\cifar-10-batches-bin'
    nb_classes = 10
    nb_samples_train = 50000
    nb_samples_eval = 10000


def parse_fn(example_serialized, is_train, img_size):
    """
    :param example_serialized:
    :param is_train:
    :return:
        image
        label: one-hot tensor
    """
    record = tf.decode_raw(example_serialized, tf.uint8)
    label = tf.slice(record, [0], [Param.LABEL_BYTES])
    label = tf.one_hot(tf.reshape(label, []), Param.nb_classes)
    image = tf.slice(record, [Param.LABEL_BYTES], [Param.IMAGE_BYTES])
    image = tf.reshape(image,
                       [Param.IMAGE_CHN, Param.IMAGE_HEI, Param.IMAGE_WID])
    image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
    image = (image - Param.IMAGE_AVE) / Param.IMAGE_STD
    if is_train:
        image = tf.image.resize_image_with_crop_or_pad(
            image, Param.IMAGE_HEI + 8, Param.IMAGE_WID + 8)
        image = tf.random_crop(
            image, [Param.IMAGE_HEI, Param.IMAGE_WID, Param.IMAGE_CHN])
        image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, img_size)
    return image, label


class Cifar10Dataset:
    """
    CIFAR-10 dataset.
        file_pattern
        dataset_fn
        parse_fn
    """

    def __init__(self, is_train, img_size):
        assert Param.data_dir is not None, '<data_dir> must not be None'
        if is_train:
            self.file_pattern = os.path.join(Param.data_dir,
                                             'data_batch_*.bin')
        else:
            self.file_pattern = os.path.join(Param.data_dir, 'test_batch.bin')

        self.dataset_fn = lambda x: tf.data.FixedLengthRecordDataset(
            x, Param.RECORD_BYTES)
        self.parse_fn = lambda x: parse_fn(x, is_train, img_size)

    def build(self, batch_size=1):
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

    def get_nb_samples(self, is_train):
        if is_train:
            return Param.nb_samples_train
        else:
            return Param.nb_samples_eval
