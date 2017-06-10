#coding=utf-8
from collections import namedtuple
from vgg import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2



HParams = namedtuple('HParams',
                     'lr_rate, mode'
                     )


class Image_zoom_net(object):
    '''Image zoom net model'''

    def __init__(self,hps, image, state_input, labels, history_vector):

        self.hps = hps
        self._image = image
        self._history_vector = history_vector
        self._labels = labels
        self._state_input = state_input



    def build_graph(self):

        self.global_step = tf.contrib.framework.get_or_create_global_step()

        self._build_model()
        if self.hps.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()



    def _build_model(self):
        '''build the model of the graph'''


        with tf.variable_scope('init'):
            x = self._image
            x_his = self._history_vector
            labels = self._labels
            state_input = self._state_input


        _,end_points = vgg_16(x,spatial_squeeze=False)
            # img_fea = vgg.vgg16.poo5(x)

        with tf.variable_scope('state'):
            im_fea_1d = tf.reshape(end_points['vgg_16/pool5'],[-1])
            self.state_net = tf.concat([im_fea_1d,x_his],axis=0)


        with tf.variable_scope('Q_net'):
            with slim.arg_scope([slim.conv2d,slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0,0.01)):
                net = slim.fully_connected(state_input,1024)
                net = slim.dropout(net,keep_prob=0.8)
                net = slim.fully_connected(net,1024)
                net = slim.dropout(net,keep_prob=0.8)
                self.qval_net = slim.fully_connected(net,10,activation_fn=None)

        with tf.variable_scope('cost'):
            self.cost = tf.losses.mean_squared_error(labels,self.qval_net)
            tf.summary.scalar('loss', self.cost)

    def _build_train_op(self):
        self.lr_rate = tf.constant(self.hps.lr_rate,tf.float32)
        tf.summary.scalar('learning_rate',self.lr_rate)
        train_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] == 'Q_net']
        #grads = tf.gradients(self.cost,train_var_list)
        optimizer = tf.train.AdamOptimizer(self.lr_rate)
        self.train_op = optimizer.minimize(self.cost,var_list=train_var_list)
        #print(self.train_op)

        # apply_op = optimizer.apply_gradients(zip(grads, train_var_list),global_step=self.global_step, name='train_step')
        # train_ops = [apply_op]
        #
        # self.train_op = tf.group(*train_ops)









