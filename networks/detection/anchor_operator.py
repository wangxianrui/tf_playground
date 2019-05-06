from .bbox_operator import *
from parameters import DetectParam as Param


def anchors_one_layer(image_size, layer_shape, anchor_size, anchor_ratio,
                      offset):
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
    x_layer, y_layer = tf.meshgrid(tf.range(layer_shape[0]),
                                   tf.range(layer_shape[1]))
    x_image = (tf.to_float(x_layer) + offset) / layer_shape[0]
    y_image = (tf.to_float(y_layer) + offset) / layer_shape[1]
    x_image = tf.expand_dims(x_image, -1)
    y_image = tf.expand_dims(y_image, -1)
    # w, h
    w_image = [
        anchor_size[0] / image_size[0],
        tf.sqrt(tf.to_float(anchor_size[0] * anchor_size[1])) / image_size[0]
    ]
    h_image = [
        anchor_size[0] / image_size[1],
        tf.sqrt(tf.to_float(anchor_size[0] * anchor_size[1])) / image_size[1]
    ]
    for ratio in anchor_ratio:
        w_image.append(w_image[0] * tf.sqrt(tf.to_float(ratio)))
        h_image.append(h_image[0] / tf.sqrt(tf.to_float(ratio)))
        w_image.append(w_image[0] / tf.sqrt(tf.to_float(ratio)))
        h_image.append(h_image[0] * tf.sqrt(tf.to_float(ratio)))
    return x_image, y_image, tf.convert_to_tensor(
        w_image), tf.convert_to_tensor(h_image)


def anchors_recheck(all_anchors):
    checked_anchors = []
    for anchors in all_anchors:
        centerx, centery, w, h = anchors
        xmin, ymin, xmax, ymax = center2point(centerx, centery, w, h)
        xmin = tf.clip_by_value(tf.reshape(xmin, [-1]), 0, 1)
        ymin = tf.clip_by_value(tf.reshape(ymin, [-1]), 0, 1)
        xmax = tf.clip_by_value(tf.reshape(xmax, [-1]), 0, 1)
        ymax = tf.clip_by_value(tf.reshape(ymax, [-1]), 0, 1)
        centerx, centery, w, h = point2center(xmin, ymin, xmax, ymax)
        checked_anchors.append(tf.stack([centerx, centery, w, h], -1))
    return tf.concat(checked_anchors, 0)


def anchors_all_layers():
    '''
    :return:
        all_anchors list [centerx, centery, w, h]
    '''
    all_anchors = []
    for i in range(len(Param.layer_shape)):
        anchors = anchors_one_layer(Param.image_size, Param.layer_shape[i],
                                    Param.anchor_size[i],
                                    Param.anchor_ratio[i], Param.offset)
        all_anchors.append(anchors)
    return anchors_recheck(all_anchors)


def do_matching(iou_matrix):
    '''
        iou_matrix: nb_gt * nb_anchors
    '''
    # match from anchor's side
    anchors_to_gt = tf.argmax(iou_matrix, axis=0)
    match_values = tf.reduce_max(iou_matrix, axis=0)
    less_mask = tf.less(match_values, Param.matching_low)
    between_mask = tf.logical_and(
        tf.less(match_values, Param.matching_high),
        tf.greater_equal(match_values, Param.matching_low))
    negative_mask = less_mask if Param.matching_ignore else between_mask
    ignore_mask = between_mask if Param.matching_ignore else less_mask
    # fill all negative positions with -1, all ignore positions is -2
    match_indices = tf.where(negative_mask, -1 * tf.ones_like(anchors_to_gt),
                             anchors_to_gt)
    match_indices = tf.where(ignore_mask, -2 * tf.ones_like(match_indices),
                             match_indices)

    anchors_to_gt_mask = tf.one_hot(tf.clip_by_value(
        match_indices, -1, tf.cast(tf.shape(iou_matrix)[0], tf.int64)),
        tf.shape(iou_matrix)[0],
        on_value=1,
        off_value=0,
        axis=0,
        dtype=tf.int32)

    # match from ground truth's side
    gt_to_anchors = tf.argmax(iou_matrix, axis=1)
    # the max match from ground truth's side has higher priority
    gt_to_anchors_mask = tf.one_hot(gt_to_anchors,
                                    tf.shape(iou_matrix)[1],
                                    on_value=1,
                                    off_value=0,
                                    axis=1,
                                    dtype=tf.int32)
    gt_to_anchors_scores = iou_matrix * tf.to_float(gt_to_anchors_mask)
    # merge matching results from ground truth's side with the original matching results from anchors' side
    # then select all the overlap score of those matching pairs
    selected_scores = tf.gather_nd(
        iou_matrix,
        tf.stack([
            tf.where(
                tf.reduce_max(gt_to_anchors_mask, axis=0) > 0,
                tf.argmax(gt_to_anchors_scores, axis=0), anchors_to_gt),
            tf.range(tf.cast(tf.shape(iou_matrix)[1], tf.int64))
        ],
            axis=1))
    # return the matching results for both foreground anchors and background anchors, also with overlap scores
    return tf.where(
        tf.reduce_max(gt_to_anchors_mask, axis=0) > 0,
        tf.argmax(gt_to_anchors_scores, axis=0),
        match_indices), selected_scores


def encode_groundtruch(ground_truth, all_anchors):
    '''
    :param ground_truth: batch * T * 6 (flags, bbox, label)
    :return:
    '''
    def encode_fn(per_truth):
        anchor_cx, anchor_cy, anchor_w, anchor_h = [tf.squeeze(
            anchor, axis=-1) for anchor in tf.split(all_anchors, 4, axis=1)]
        gt_flags, gt_bboxes, gt_labels = tf.split(
            per_truth, [1, 4, 1], axis=-1)
        gt_flags = tf.squeeze(gt_flags, axis=-1)
        gt_labels = tf.squeeze(gt_labels, axis=-1)
        index = tf.where(gt_flags > 0)
        gt_bboxes = tf.gather_nd(gt_bboxes, index)
        gt_labels = tf.gather_nd(gt_labels, index)
        # nums_gt * nums_anchors
        iou_matrix = bbox_iou(gt_bboxes, all_anchors)
        matched_gt, gt_scores = do_matching(iou_matrix)
        matched_gt_mask = matched_gt > -1
        matched_indices = tf.clip_by_value(matched_gt, 0, tf.int64.max)
        gt_labels = tf.cast(tf.gather(gt_labels, matched_indices), tf.int64)
        gt_labels = gt_labels * tf.cast(matched_gt_mask, tf.int64)
        gt_labels = gt_labels + (-1 * tf.cast(matched_gt < -1, tf.int64))
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = tf.unstack(
            tf.gather(gt_bboxes, matched_indices), 4, axis=-1)
        gt_cx, gt_cy, gt_w, gt_h = point2center(
            gt_xmin, gt_ymin, gt_xmax, gt_ymax)
        gt_cx = (gt_cx - anchor_cx) / anchor_w / Param.prior_scaling[0]
        gt_cy = (gt_cy - anchor_cy) / anchor_h / Param.prior_scaling[0]
        gt_w = tf.log(gt_w / anchor_w) / Param.prior_scaling[1]
        gt_h = tf.log(gt_h / anchor_h) / Param.prior_scaling[1]
        target_location = tf.stack([gt_cx, gt_cy, gt_w, gt_h], axis=-1)
        target_location = tf.expand_dims(
            tf.cast(matched_gt_mask, tf.float32), -1) * target_location
        return target_location, gt_labels, gt_scores

    # target_location, target_label, target_score = encode_fn(ground_truth[0])
    target_location, target_label, target_score = tf.map_fn(
        encode_fn, ground_truth, dtype=(tf.float32, tf.int64, tf.float32), back_prop=False)
    return target_location, target_label, target_score


def decode_prediction():
    pass
