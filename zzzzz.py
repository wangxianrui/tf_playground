import tensorflow as tf

from datasets.pascalvoc_dataset import PascalVocDataset
from networks.detection.peleenet_ssd import PeleeNetSSD
from parameters import DetectParam as Param

sess = tf.Session()

dataset = PascalVocDataset(True, Param.image_size)
img_info, images, groundtruth = dataset.build(Param.batch_size).get_next()

pelee_ssd = PeleeNetSSD()
locations, classes = pelee_ssd.forward(images, True)
pelee_ssd.calc_loss(locations, classes, groundtruth)
