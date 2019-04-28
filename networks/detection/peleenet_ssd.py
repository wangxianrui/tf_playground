from networks.classify.peleenet import PeleeNet
from .anchor_operator import *


class PeleeNetSSD:
    def __init__(self):
        self.peleenet = PeleeNet()
        self.extra_output_channel = 256

    def conv_bn_relu(self,
                     inputs,
                     is_training,
                     output_channel,
                     kernel_size,
                     stride,
                     padding='same',
                     use_relu=True):
        conv = tf.layers.Conv2D(
            output_channel, kernel_size, stride, padding,
            use_bias=False)(inputs)
        conv_bn = tf.layers.BatchNormalization()(conv, training=is_training)
        if use_relu:
            return tf.nn.relu(conv_bn)
        else:
            return conv_bn

    def add_extra(self, inputs, is_training, output_channel):
        a2 = self.conv_bn_relu(
            inputs, is_training, output_channel, 1, 1, use_relu=False)
        b2a = self.conv_bn_relu(inputs, is_training, int(output_channel / 2),
                                1, 1)
        b2b = self.conv_bn_relu(b2a, is_training, int(output_channel / 2), 3,
                                1)
        b2c = self.conv_bn_relu(
            b2b, is_training, output_channel, 1, 1, use_relu=False)
        return a2 + b2c

    def forward(self, inputs, is_train):
        with tf.variable_scope('backbone'):
            stage3, stage4 = self.peleenet.forward(inputs, is_train)

        with tf.variable_scope('feature_exactor'):
            feature_layers = []
            with tf.variable_scope('extra_pm2'):
                pm2_inputs = stage3
                pm2_res = self.add_extra(pm2_inputs, is_train,
                                         self.extra_output_channel)
                feature_layers.append(pm2_res)
                feature_layers.append(pm2_res)

            with tf.variable_scope('extra_pm3'):
                pm3_inputs = stage4
                pm3_res = self.add_extra(pm3_inputs, is_train,
                                         self.extra_output_channel)
                feature_layers.append(pm3_res)

            with tf.variable_scope('extra_pm3_to_pm4'):
                pm3_to_pm4 = tf.layers.Conv2D(self.extra_output_channel, 1,
                                              1)(pm3_inputs)
                pm3_to_pm4 = tf.nn.relu(pm3_to_pm4)
                pm3_to_pm4 = tf.layers.Conv2D(
                    self.extra_output_channel, 3, 2,
                    padding='same')(pm3_to_pm4)
                pm3_to_pm4 = tf.nn.relu(pm3_to_pm4)

            with tf.variable_scope('extra_pm4'):
                pm4_inputs = pm3_to_pm4
                pm4_res = self.add_extra(pm4_inputs, is_train,
                                         self.extra_output_channel)
                feature_layers.append(pm4_res)

            with tf.variable_scope('extra_pm4_to_pm5'):
                pm4_to_pm5 = tf.layers.Conv2D(self.extra_output_channel, 1,
                                              1)(pm4_inputs)
                pm4_to_pm5 = tf.nn.relu(pm4_to_pm5)
                pm4_to_pm5 = tf.layers.Conv2D(self.extra_output_channel, 3,
                                              1)(pm4_to_pm5)
                pm4_to_pm5 = tf.nn.relu(pm4_to_pm5)

            with tf.variable_scope('extra_pm5'):
                pm5_inputs = pm4_to_pm5
                pm5_res = self.add_extra(pm5_inputs, is_train,
                                         self.extra_output_channel)
                feature_layers.append(pm5_res)

            with tf.variable_scope('extra_pm5_to_pm6'):
                pm5_to_pm6 = tf.layers.Conv2D(self.extra_output_channel, 1,
                                              1)(pm5_inputs)
                pm5_to_pm6 = tf.nn.relu(pm5_to_pm6)
                pm5_to_pm6 = tf.layers.Conv2D(self.extra_output_channel, 3,
                                              1)(pm5_to_pm6)
                pm5_to_pm6 = tf.nn.relu(pm5_to_pm6)

            with tf.variable_scope('extra_pm6'):
                pm6_inputs = pm5_to_pm6
                pm6_res = self.add_extra(pm6_inputs, is_train,
                                         self.extra_output_channel)
                feature_layers.append(pm6_res)

        with tf.variable_scope('detection'):
            locations = []
            classes = []
            for i, feature in enumerate(feature_layers):
                locations.append(
                    tf.layers.Conv2D(
                        Param.anchor_depth_per_layer[i] * 4,
                        kernel_size=3,
                        padding='same',
                        use_bias=True)(feature))
                classes.append(
                    tf.layers.Conv2D(
                        Param.anchor_depth_per_layer[i] * Param.num_classes,
                        kernel_size=3,
                        padding='same',
                        use_bias=True)(feature))
            batch_size = tf.shape(inputs)[0]
            locations = [tf.reshape(loc, [batch_size, -1, 4]) for loc in locations]
            classes = [tf.reshape(cla, [batch_size, -1, Param.num_classes]) for cla in classes]

            return tf.concat(locations, 1), tf.concat(classes, 1)

    def calc_loss(self, pred_locations, pred_classes, groundtruth):
        '''
        :param pred_locations:  batch * T * 4
        :param pred_classes:    batch * T * num_classes
        :param groundtruth:     T * 6 (flags, bboxes, labels)
        :return:
        '''
        enco_location, enco_classes = encode_groundtruch(groundtruth)

    def calc_prediction(self):
        pass
