import tensorflow as tf

label_map = {
    'person': 1,
    'bicycle': 2,
    'car': 3,
    'motorcycle': 4,
    'airplane': 5,
    'bus': 6,
    'train': 7,
    'truck': 8,
    'boat': 9,
    'traffic light': 10,
    'fire hydrant': 11,
    'stop sign': 13,
    'parking meter': 14,
    'bench': 15,
    'bird': 16,
    'cat': 17,
    'dog': 18,
    'horse': 19,
    'sheep': 20,
    'cow': 21,
    'elephant': 22,
    'bear': 23,
    'zebra': 24,
    'giraffe': 25,
    'backpack': 27,
    'umbrella': 28,
    'handbag': 31,
    'tie': 32,
    'suitcase': 33,
    'frisbee': 34,
    'skis': 35,
    'snowboard': 36,
    'sports ball': 37,
    'kite': 38,
    'baseball bat': 39,
    'baseball glove': 40,
    'skateboard': 41,
    'surfboard': 42,
    'tennis racket': 43,
    'bottle': 44,
    'wine glass': 46,
    'cup': 47,
    'fork': 48,
    'knife': 49,
    'spoon': 50,
    'bowl': 51,
    'banana': 52,
    'apple': 53,
    'sandwich': 54,
    'orange': 55,
    'broccoli': 56,
    'carrot': 57,
    'hot dog': 58,
    'pizza': 59,
    'donut': 60,
    'cake': 61,
    'chair': 62,
    'couch': 63,
    'potted plant': 64,
    'bed': 65,
    'dining table': 67,
    'toilet': 70,
    'tv': 72,
    'laptop': 73,
    'mouse': 74,
    'remote': 75,
    'keyboard': 76,
    'cell phone': 77,
    'microwave': 78,
    'oven': 79,
    'toaster': 80,
    'sink': 81,
    'refrigerator': 82,
    'book': 84,
    'clock': 85,
    'vase': 86,
    'scissors': 87,
    'teddy bear': 88,
    'hair drier': 89,
    'toothbrush': 90
}

category = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
    88, 89, 90
]


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
