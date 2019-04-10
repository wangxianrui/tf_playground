from networks.peleenet_ssd import anchor_operator

# size or shape with dim [w, h] [x, y]
image_size = [320, 320]
# anchor
layer_shape = [38, 19, 10, 5, 3, 1]
layer_step = [8, 16, 32, 64, 100, 300]
anchor_ratio = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
min_size = [30, 60, 111, 162, 213, 264]
max_size = [60, 111, 162, 213, 264, 315]
variance = [0.1, 0.2]
offset = 0.5

all_anchors = anchor_operator.get_all_anchors(image_size, layer_shape, min_size, max_size, anchor_ratio, offset)

for anchors in all_anchors:
    for i in anchors:
        print(i)
    input()
