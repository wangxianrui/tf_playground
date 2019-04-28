# classify
class ClassifyParam:
    # size or shape with dim [w, h] [x, y]
    image_size = [32, 32]
    num_classes = 10
    batch_size = 64
    summary_step = 500
    save_step = 1e4
    save_path = 'checkpoint'

    #
    lr_val = [0.01, 0.001, 0.00001]
    lr_step = [int(1e4 * step) for step in [5, 10]]
    max_step = int(1e4 * 15)
    momentum = 0.9
    weight_decay = 1e-4


class DetectParam:
    # size or shape with dim [w, h] [x, y]
    image_size = [320, 320]
    num_classes = 21
    batch_size = 4
    summary_op = 500
    save_step = 1e4

    #
    lr_val = [0.1, 0.001, 0.00001]
    lr_step = [1e4 * step for step in [5, 10]]
    max_step = 1e4 * 15
    momentum = 0.9
    weight_decay = 1e-4

    # shape with list [w, h] or integer wh
    layer_shape = [38, 19, 10, 5, 3, 1]
    layer_step = [8, 16, 32, 64, 100, 300]
    anchor_depth_per_layer = [6, 6, 6, 6, 6, 6]
    anchor_ratio = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    min_size = [30, 60, 111, 162, 213, 264]
    max_size = [60, 111, 162, 213, 264, 315]
    variance = [0.1, 0.2]
    offset = 0.5
    match_low = 0.3
    match_high = 0.7
    match_ignore = True
