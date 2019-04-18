import numpy as np


def get_layer_anchors(image_size, layer_shape, min_size, max_size,
                      anchor_ratio, offset):
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if isinstance(layer_shape, int):
        layer_shape = (layer_shape, layer_shape)
    # x, y
    x_layer, y_layer = np.meshgrid(
        range(layer_shape[0]), range(layer_shape[1]))
    x_image = (x_layer + offset) / layer_shape[0]
    y_image = (y_layer + offset) / layer_shape[1]
    x_image = np.expand_dims(x_image, -1)
    y_image = np.expand_dims(y_image, -1)
    # w, h
    w_image = [min_size / image_size[0], max_size / image_size[0]]
    h_image = [min_size / image_size[1], max_size / image_size[1]]
    for ratio in anchor_ratio:
        w_image.append(w_image[0] * np.sqrt(ratio))
        h_image.append(h_image[0] / np.sqrt(ratio))
        w_image.append(w_image[0] / np.sqrt(ratio))
        h_image.append(h_image[0] * np.sqrt(ratio))
    return x_image, y_image, np.array(w_image), np.array(h_image)


def get_all_anchors(image_size, layer_shape, min_size, max_size, anchor_ratio,
                    offset):
    all_anchors = []
    for i in range(len(layer_shape)):
        layer_anchors = get_layer_anchors(image_size, layer_shape[i],
                                          min_size[i], max_size[i],
                                          anchor_ratio[i], offset)
        all_anchors.append(layer_anchors)
    return all_anchors


def encode_anchors():
    pass


def decode_anchors():
    pass
