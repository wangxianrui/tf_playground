import sys
sys.path.append('../')

from parameters import DetectParam as Param
from networks.detection.anchor_operator import anchor_operator

all_anchors = anchor_operator.get_all_anchors(
    Param.image_size, Param.layer_shape, Param.min_size, Param.max_size,
    Param.anchor_ratio, Param.offset)

for anchors in all_anchors:
    for i in anchors:
        print(i)
    input()
