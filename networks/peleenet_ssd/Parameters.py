class Param:
    # size or shape with dim [w, h] [x, y]
    image_size = [320, 320]
    # anchor
    layer_shape = [38, 19, 10, 5, 3, 1]
    layer_step = [8, 16, 32, 64, 100, 300]
    anchor_ratio = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    min_size = [30, 60, 111, 162, 213, 264]
    max_size = [60, 111, 162, 213, 264, 315]
    variance = [0.1, 0.2]