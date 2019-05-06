import time

import tensorflow as tf
from tqdm import trange

from datasets.pascalvoc_dataset import PascalVocDataset
from networks.detection.peleenet_ssd import PeleeNetSSD
from parameters import DetectParam as Param


def warm_start():
    pass


def train():
    # sess
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # data
    dataset = PascalVocDataset(True, Param.image_size).build(Param.batch_size)
    img_info, image, groundtruth = dataset.get_next()

    # model
    model = PeleeNetSSD()
    pred_locations, pred_classes = model.forward(image, is_train=True)
    loss = model.calc_loss(pred_locations, pred_classes, groundtruth,
                           tf.trainable_variables())
    sess.run(loss)
    exit()
    acc_top1, acc_top5 = lenet.calc_accuracy(logits, labels)

    # optimizer
    global_step = tf.train.get_or_create_global_step()
    lrn_rate = tf.train.piecewise_constant(global_step, Param.lr_step,
                                           Param.lr_val)
    optimizer = tf.train.AdamOptimizer(lrn_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    # prepare
    logdir = 'log/' + str(time.time()) + '/'
    sm_writer = tf.summary.FileWriter(logdir, sess.graph)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc_top1', acc_top1)
    tf.summary.scalar('acc_top5', acc_top5)
    tf.summary.scalar('lrn_rate', lrn_rate)
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # run
    sess.run(init_op)
    warm_start()
    for index in trange(Param.max_step):
        sess.run(train_op)
        if (index + 1) % Param.summary_step == 0:
            summary = sess.run(summary_op)
            sm_writer.add_summary(summary, index)
        if (index + 1) % Param.save_step == 0:
            saver.save(sess, 'checkpoint/model.ckpt', global_step)


train()
