import tensorflow as tf

from parameters import DetectParam as Param


def areas(bboxes):
    xmin, ymin, xmax, ymax, = tf.split(bboxes, 4, axis=1)
    return (xmax - xmin) * (ymax - ymin)


def intersection(bboxes1, bboxes2):
    # T x 1
    xmin_1, ymin_1, xmax_1, ymax_1 = tf.split(bboxes1, 4, axis=1)
    # 1 x T
    xmin_2, ymin_2, xmax_2, ymax_2 = tf.split(tf.transpose(bboxes2, [1, 0]), 4, axis=0)
    # broadcast here to generate the full matrix
    int_xmin = tf.maximum(xmin_1, xmin_2)
    int_ymin = tf.maximum(ymin_1, ymin_2)
    int_xmax = tf.minimum(xmax_1, xmax_2)
    int_ymax = tf.minimum(ymax_1, ymax_2)
    w = tf.maximum(int_xmax - int_xmin, 0.)
    h = tf.maximum(int_ymax - int_ymin, 0.)
    return h * w


def bbox_iou(bboxes1, bboxes2):
    '''
    :param bboxes1: T1 * 4
    :param bboxes2: T2 * 4
    :return:  T1 * T2
    '''
    # T1 * T2
    inter_vol = intersection(bboxes1, bboxes2)
    union_vol = areas(bboxes1) + tf.transpose(areas(bboxes2), perm=[1, 0]) - inter_vol
    return tf.where(tf.equal(union_vol, 0.0), tf.zeros_like(inter_vol), inter_vol / union_vol)


def center2point(centerx, centery, width, height):
    xmin = centerx - width / 2.0
    ymin = centery - height / 2.0
    xmax = centerx + width / 2.0
    ymax = centery + height / 2.0
    return xmin, ymin, xmax, ymax


def point2center(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    centerx = xmin + width / 2.0
    centery = ymin + height / 2.0
    return centerx, centery, width, height


def bbox_matching(iou_matrix):
    anchors_to_gt = tf.argmax(iou_matrix, axis=0)
    match_values = tf.reduce_max(iou_matrix, axis=0)
    less_mask = tf.less(match_values, Param.match_low)
    between_mask = tf.logical_and(tf.less(match_values, Param.match_high),
                                  tf.greater_equal(match_values, Param.match_low))
    negative_mask = less_mask if Param.match_ignore else between_mask
    ignore_mask = between_mask if Param.match_ignore else less_mask
    match_indices = tf.where(negative_mask, -1 * tf.ones_like(anchors_to_gt), anchors_to_gt)
    match_indices = tf.where(ignore_mask, -2 * tf.ones_like(match_indices), match_indices)
    anchors_to_gt_mask = tf.one_hot(tf.clip_by_value(match_indices, -1, tf.cast(tf.shape(iou_matrix)[0], tf.int64)),
                                    tf.shape(iou_matrix)[0], on_value=1, off_value=0, axis=0, dtype=tf.int32)
    # match from ground truth's side
    gt_to_anchors = tf.argmax(iou_matrix, axis=1)
    left_gt_to_anchors_mask = tf.one_hot(gt_to_anchors, tf.shape(iou_matrix)[1], on_value=1, off_value=0, axis=1,
                                         dtype=tf.int32)
    left_gt_to_anchors_scores = iou_matrix * tf.to_float(left_gt_to_anchors_mask)
    selected_scores = tf.gather_nd(iou_matrix,
                                   tf.stack([tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                                                      tf.argmax(left_gt_to_anchors_scores, axis=0),
                                                      anchors_to_gt),
                                             tf.range(tf.cast(tf.shape(iou_matrix)[1], tf.int64))], axis=1))
    return tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                    tf.argmax(left_gt_to_anchors_scores, axis=0),
                    match_indices), selected_scores
