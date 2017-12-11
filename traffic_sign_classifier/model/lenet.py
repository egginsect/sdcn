from .basemodel import BaseModel
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import flatten
class LeNet(BaseModel):
    def buildGraph(self):
        self.mu = 0
        self.sigma = 0.1
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.5)):
            net = slim.conv2d(self.inputs['images'], 6, [5, 5], scope='conv1', padding='VALID')
            net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='VALID')
            #net = self.dropout(net, 'dropout1') 
            net = slim.conv2d(net, 16, [5, 5], scope='conv2', padding='VALID')
            net = slim.max_pool2d(net, [2, 2], scope='pool2', padding='VALID')
            net = self.dropout(net, 'dropout2') 
            net = slim.conv2d(net, 32, [1, 1], scope='conv3', padding='VALID')
            #net = slim.max_pool2d(net, [2, 2], scope='pool3', padding='VALID')
            net = self.dropout(net, 'dropout3') 
            net = slim.flatten(net)
            net = slim.fully_connected(net, 120, scope='fc1')
            net = self.dropout(net, 'dropout4') 
            net = slim.fully_connected(net, 84, scope='fc2')
            self.logits = slim.fully_connected(net, 43, activation_fn=None, scope='logits')

class LeNetExt1(BaseModel):
    def buildGraph(self):
        self.mu = 0
        self.sigma = 0.1
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      #weights_regularizer=slim.l2_regularizer(0.005)
                      ):
            net = slim.conv2d(self.inputs['images'], 6, [5, 5], scope='conv1', padding='VALID')
            net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='SAME')
            #net = self.dropout(net, 'dropout1') 
            net = slim.conv2d(net, 16, [5, 5], scope='conv2', padding='SAME')
            #net = self.dropout(net, 'dropout2') 
            net = slim.conv2d(net, 16, [3, 3], scope='conv3', padding='SAME')
            net = slim.max_pool2d(net, [2, 2], scope='pool3', padding='SAME')
            net = self.dropout(net, 'dropout3') 
            net = slim.conv2d(net, 32, [3, 3], scope='conv4', padding='SAME')
            net = slim.conv2d(net, 64, [1, 1], scope='conv5', padding='SAME')
            net = self.dropout(net, 'dropout4') 
            net = slim.flatten(net)
            net = slim.fully_connected(net, 120, scope='fc1')
            net = self.dropout(net, 'dropout4') 
            net = slim.fully_connected(net, 120, scope='fc2')
            self.logits = slim.fully_connected(net, 43, activation_fn=None, scope='logits')
class LeNetExt2(BaseModel):
    def buildGraph(self):
        self.mu = 0
        self.sigma = 0.1
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.selu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(1e-3)
                      ):
            net = slim.conv2d(self.inputs['images'], 6, [5, 5], scope='conv1', padding='VALID')
            net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='SAME')
            #net = self.dropout(net, 'dropout1') 
            net = slim.conv2d(net, 16, [5, 5], scope='conv2', padding='SAME')
            #net = self.dropout(net, 'dropout2') 
            net = slim.conv2d(net, 32, [3, 3], scope='conv3', padding='SAME')
            net = slim.max_pool2d(net, [2, 2], scope='pool3', padding='SAME')
            net = self.dropout(net, 'dropout3') 
            net = slim.conv2d(net, 32, [3, 3], scope='conv4', padding='SAME')
            net = slim.conv2d(net, 64, [3, 3], scope='conv5', padding='SAME')
            net = self.dropout(net, 'dropout4') 
            net = slim.flatten(net)
            net = slim.fully_connected(net, 120, scope='fc1')
            net = self.dropout(net, 'dropout4') 
            net = slim.fully_connected(net, 120, scope='fc2')
            self.logits = slim.fully_connected(net, 43, activation_fn=None, scope='logits')
