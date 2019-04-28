import numpy as np
import tensorflow as tf
from tqdm import trange

from datasets.cifar10_dataset import Cifar10Dataset
from networks.classify.lenet import LeNet
from parameters import ClassifyParam as Param


def warm_start(sess, saver):
    save_path = tf.train.latest_checkpoint(Param.save_path)
    saver.restore(sess, save_path)


def evaluate():
    # sess
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # data
    dataset = Cifar10Dataset(True, Param.image_size)
    images, labels = dataset.build().get_next()

    # model
    lenet = LeNet()
    logits = lenet.forward(images, False)
    acc_top1, acc_top5 = lenet.calc_accuracy(logits, labels)

    # prepare
    saver = tf.train.Saver()

    # run
    warm_start(sess, saver)
    nb_samples = dataset.get_nb_samples(is_train=False)
    accuracy = np.zeros([nb_samples, 2])
    for index in trange(nb_samples):
        accuracy[index] = sess.run([acc_top1, acc_top5])
        if (index + 1) % Param.summary_step == 0:
            print('acc_top1:{}, acc_top5:{}'.
                  format(np.mean(accuracy[:index, 0]), np.mean(accuracy[:index, 1])))
    print('acc_top1:{}, acc_top5:{}'.format(np.mean(accuracy[:, 0]), np.mean(accuracy[:, 1])))


evaluate()
