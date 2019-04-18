import tensorflow as tf
from parameters import Param
from datasets.pascalvoc_dataset import PascalVocDataset
from networks.peleenet_ssd.peleenet_ssd import PeleeNetSSD


def warm_start():
    pass


def train():
    # sess
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # data
    dataset = PascalVocDataset(
        is_train=True, Param.img_size).build(Param.batch_size)
    img_info, images, groundtruth = dataset.get_next()

    # model
    pelee_ssd = PeleeNetSSD(Param.num_classes)
    prediction = pelee_ssd.forward(images, is_train=True)
    loss = pelee_ssd.calc_loss()

    # optimizer
    global_step = tf.train.get_or_create_global_step()
    lrn_rate = tf.train.piecewise_constant(global_step, Param.lr_step,
                                           Param.lr_val)
    optimizer = tf.train.MomentumOptimizer(lrn_rate, Param.momentum)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    # prepare
    init = tf.global_variables_initializer()

    # run
    sess.run(init)
    warm_start()
