from collections import namedtuple
from vgg import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2



class Q_net():
    def __init__(self,name):
        self.state_input = tf.placeholder(tf.float32,[None,25128],name='input_state')
        self.global_step = tf.placeholder(tf.int32,[])
        with tf.variable_scope('Q_net'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
                net = slim.fully_connected(self.state_input, 1024)
                net = slim.dropout(net, keep_prob=0.5)
                net = slim.fully_connected(net, 1024)
                net = slim.dropout(net, keep_prob=0.5)
                streamA, streamV = tf.split(net, 2, 1)
                AW = slim.fully_connected(streamA, 10,activation_fn=None)
                VW = slim.fully_connected(streamV, 1,activation_fn=None)
                self.qval = VW + tf.subtract(AW, tf.reduce_mean(AW, reduction_indices=1, keep_dims=True))
                # max Q value
                # self.qval = slim.fully_connected(net,10)
                self.predict = tf.argmax(self.qval, 1)


        self.target_Q = tf.placeholder(tf.float32,[None,10])
        self.cost = tf.losses.mean_squared_error(self.target_Q, self.qval)
        lr_rate = 3e-7
        #self.lr_rate = tf.train.exponential_decay(lr_rate,global_step=self.global_step,decay_steps=2,decay_rate=0.9)
        tf.summary.scalar('learning_rate', lr_rate)
        # train_var = tf.trainable_variables()
        train_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] == name]


        learning_rate = tf.train.exponential_decay(learning_rate=lr_rate,
                                                   global_step=self.global_step,decay_steps=1,decay_rate=0.9)
        self.lr_rate = learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.cost, var_list=train_var_list)
        tf.summary.scalar('loss', self.cost)

class vgg_net():
    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_img')
        _, end_points = vgg_16(self.image, spatial_squeeze=False)
        self.imgfea = end_points['vgg_16/pool5']

def updateTargetGraph(tfVars,tau):
    tfVar_main = [v for v in tfVars if v.name.split('/')[0] == 'mainQN']
    tfVar_target = [v for v in tfVars if v.name.split('/')[0] == 'targetQN']
    op_holder = []
    for idx, var in enumerate(tfVar_main):
        op_holder.append(tfVar_target[idx].assign(var.value()*tau +((1-tau)*tfVar_target[idx].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


#mainQN = Q_net('mainQN')

def get_weights():
    #a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]