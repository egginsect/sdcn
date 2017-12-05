from .basemodel import BaseModel
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import flatten
class VGG16(BaseModel):
    def buildGraph(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(self.inputs['images'], 2, slim.conv2d, 4, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = self.dropout(net, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = self.dropout(net, scope='dropout7')
            net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
            self.logits = slim.fully_connected(net, 43, activation_fn=None, scope='logits')
        return net

class VGGsimple(BaseModel):
    def buildGraph(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.005)):
            net = slim.conv2d(self.inputs['images'], 4, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 8, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net, 16, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.conv2d(net, 16, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.conv2d(net, 64, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = self.dropout(net, scope='dropout5')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024, scope='fc6')
            net = self.dropout(net, scope='dropout6')
            net = slim.fully_connected(net, 128, activation_fn=None, scope='fc8')
            self.logits = slim.fully_connected(net, 43, activation_fn=None, scope='logits')
        return net
