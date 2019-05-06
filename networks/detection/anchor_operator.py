from .bbox_operator import *

def get_layer_anchors(image_size, layer_shape, min_size, max_size, anchor_ratio, offset):
    '''
    :return:
        x_image, y_image  with the shape of layer_shape,
        w_image, h_image  with the shape of anchor_ratio
    '''
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if isinstance(layer_shape, int):
        layer_shape = (layer_shape, layer_shape)
    # x, y
    x_layer, y_layer = tf.meshgrid(tf.range(layer_shape[0]), tf.range(layer_shape[1]))
    x_image = (tf.to_float(x_layer) + offset) / layer_shape[0]
    y_image = (tf.to_float(y_layer) + offset) / layer_shape[1]
    x_image = tf.expand_dims(x_image, -1)
    y_image = tf.expand_dims(y_image, -1)
    # w, h
    w_image = [min_size / image_size[0], max_size / image_size[0]]
    h_image = [min_size / image_size[1], max_size / image_size[1]]
    for ratio in anchor_ratio:
        w_image.append(w_image[0] * tf.sqrt(tf.to_float(ratio)))
        h_image.append(h_image[0] / tf.sqrt(tf.to_float(ratio)))
        w_image.append(w_image[0] / tf.sqrt(tf.to_float(ratio)))
        h_image.append(h_image[0] * tf.sqrt(tf.to_float(ratio)))
    return x_image, y_image, tf.convert_to_tensor(w_image), tf.convert_to_tensor(h_image)


def get_all_anchors(image_size, layer_shape, min_size, max_size, anchor_ratio, offset):
    '''
    :return:
        all_anchors with shape T * 4
    '''
    all_anchors = []
    for i in range(len(layer_shape)):
        centerx, centery, width, height = get_layer_anchors(image_size, layer_shape[i], min_size[i],
                                                            max_size[i], anchor_ratio[i], offset)
        anchors = center2point(centerx, centery, width, height)
        anchors = tf.stack(anchors, axis=-1)
        all_anchors.append(tf.reshape(anchors, [-1, 4]))
    all_anchors = tf.concat(all_anchors, 0)
    return all_anchors


def encode_fn(per_truth, all_anchors):
    '''

    :param per_truth:  ground_truth on one image T1 * 6
    :param all_anchors:  T2 * 4
    :return:
    '''
    gt_flags, gt_bboxes, gt_labels = tf.split(per_truth, [1, 4, 1], axis=-1)
    gt_flags = tf.squeeze(gt_flags, axis=-1)
    index = tf.where(gt_flags > 0)
    gt_bboxes, gt_labels = tf.gather_nd(gt_bboxes, index), tf.gather_nd(gt_labels, index)
    # nums_gt * nums_anchors
    iou_matrix = bbox_iou(gt_bboxes, all_anchors)

    # return enco_loc, enco_cls, enco_score


def encode_groundtruch(ground_truth):
    '''
    :param ground_truth: batch * T * 6 (flags, bbox, label)
    :return:
    '''
    # T * 4
    all_anchors = get_all_anchors()
    # tf.map_fn(encode_fn, groundtruth)
    encode_fn(ground_truth[0], all_anchors)


def decode_prediction():
    pass
