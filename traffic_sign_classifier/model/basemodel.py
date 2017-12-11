import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import sys
import glob
import time
from collections import defaultdict
import math
import numpy as np
import pdb
class BaseModel(object):
    def __init__(self, config, label_names=None):
        self.label_names = label_names
        self.config = config
        self.createExprCase()
        self.createInputs()
        self.buildGraph()
        self.createLoss()
        self.createOptimizer()
        self.createMetrics()
        self.createSummary()
        self.setSession()

    def createExprCase(self):
        if self.config.exprdir:
            cleaned_path = glob.glob(self.config.exprdir)[0]
            self.logdir = os.path.dirname(cleaned_path) 
            self.exprcase = os.path.basename(cleaned_path)
        else:
            self.logdir = self.config.logdir
            self.exprcase = self.__class__.__name__+time.strftime("-%Y-%m-%d-%H:%M")
            self.exprcase+='-{}'.format(self.config.optimizer)
            self.exprcase+='-bs{}'.format(self.config.batch_size)
            self.exprcase+='-lr{:.1e}'.format(self.config.learning_rate)

    def genFolder(self, pathname):
        path = os.path.join(self.logdir, self.exprcase, pathname)
        if not glob.glob(path):
            os.makedirs(path) 
        return path

    def dropout(self, tensor, scope):
        return tf.cond(self.inputs['isTrain'], lambda :slim.dropout(tensor, self.config.keep_prob, scope=scope),
        lambda :tf.identity(tensor)) 
        
    def setSession(self):
        summarypath = self.genFolder('tfboard')
        self.summary_writer = tf.summary.FileWriter(summarypath)
        self.saver = tf.train.Saver(max_to_keep=30)
        self.sess = tf.InteractiveSession()
        self.ckptpath = self.genFolder('ckpt')
        ckptstate = tf.train.get_checkpoint_state(self.ckptpath)
        if ckptstate:
            try:
                self.restoreModel(self.config.ckpt_number)
            except:
                raise 'Invalid checkpoint number'
        else:
            self.sess.run(tf.global_variables_initializer())
            
    def data_augmetation(self, images):
        samples = tf.squeeze(tf.multinomial(tf.fill([4,2], 10.), 1))
        def random_rotate():
            angle = tf.random_uniform([1], minval=-20, maxval=20, dtype=tf.float32)*math.pi/180
            return tf.contrib.image.rotate(images, angle)
        images = tf.cond(tf.equal(samples[0],1), random_rotate, lambda: tf.identity(images))
        random_brightness = lambda : tf.image.random_brightness(images, 10) 
        images = tf.cond(tf.equal(samples[1],1), random_brightness, lambda: tf.identity(images))
        random_contrast = lambda : tf.image.random_contrast(images, 1, 10) 
        images = tf.cond(tf.equal(samples[2],1), random_contrast, lambda: tf.identity(images))
        """
        def random_crop():
            return tf.map_fn(lambda img: tf.random_crop(tf.pad(img,\
                tf.constant([[3,3], [3, 3], [0,0]])), [32, 32, 1]), images)
        images = tf.cond(tf.equal(samples[3],1), random_crop, lambda: tf.identity(images))
        """
        return images
        
        
    def createInputs(self):
        self.inputs = {} 
        self.feeding_inputs = {
            'isTrain':tf.placeholder(tf.bool, []),
            'labels':tf.placeholder(tf.int64, (None))
        }
        if self.config.grayscale:
            self.feeding_inputs['images'] = tf.placeholder(tf.float32, (None, 32, 32, 1)) 
        else:
            self.feeding_inputs['images'] = tf.placeholder(tf.float32, (None, 32, 32, 3)) 
        self.inputs.update(self.feeding_inputs)
        center_cropped = tf.map_fn(lambda img: tf.image.central_crop(img, 0.8), self.inputs['images'])
        self.inputs['images'] = tf.image.resize_images(center_cropped, [32,32])

        if self.config.data_augmentation:

            self.inputs['images'] = tf.cond(self.inputs['isTrain'],\
            lambda: self.data_augmetation(self.inputs['images']), lambda: self.inputs['images'])
        

    def buildGraph(self):
        raise 'Implement Graph first'

    def createLoss(self):
        onehot_labels = tf.one_hot(self.inputs['labels'], 43)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=self.logits)) 


    def createOptimizer(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.max_validation_accuracy = tf.Variable(0, name='max_validation_accuracy', trainable=False)
        if self.config.adaptive_learning_rate:
            learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step,
            self.config.decay_steps, self.config.learning_rate_decay, staircase=True)
        else:
            learning_rate = self.config.learning_rate
        if self.config.optimizer == 'Momentum':
            self.optimizer = getattr(tf.train, self.config.optimizer+'Optimizer')(learning_rate = learning_rate, momentum=0.9)
        else:
            self.optimizer = getattr(tf.train, self.config.optimizer+'Optimizer')(learning_rate = learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def createMetrics(self):
        self.prediction = tf.argmax(self.logits, axis=-1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.inputs['labels'], self.prediction), tf.float32))
        self.top5, _ = tf.nn.top_k(tf.nn.softmax(self.logits), 5)

    def createSummary(self):
        with tf.name_scope('train'):
            train_summary = []
            train_summary.append(tf.summary.scalar(name='loss', tensor=self.loss))
            train_summary.append(tf.summary.scalar(name='accuracy', tensor=self.accuracy))
            self.train_summary = tf.summary.merge(train_summary)


    def trainStep(self, data):
        evaldict = {
            'summary':self.train_summary,
            'trainop':self.training_operation, 
            'loss':self.loss, 
            'accuracy':self.accuracy,
            'global_step':self.global_step,
            'max_val_accuracy':self.max_validation_accuracy
        }
        data['isTrain'] = True
        feed_dict={v:data[k] for k,v in self.feeding_inputs.items()}
        return self.sess.run(evaldict, feed_dict)
    
    def predict(self, data):
        result = self.sess.run([self.prediction, self.top5] , feed_dict={v:data[k] for k,v in self.feeding_inputs.items() if k in data})
        return [self.label_names[lab] for lab in result[0]], result[1]

    def testStep(self, testbg, validation=False):
        evaldict = {
            'loss':self.loss, 
            'accuracy':self.accuracy,
        }
        evalvals = defaultdict(list) 
        total_numdata = 0
        for item in testbg:
            item['isTrain'] = False
            feed_dict = {v:item[k] for k,v in self.feeding_inputs.items()}
            result = self.sess.run(evaldict, feed_dict)
            numdata = len(item['images'])
            total_numdata+=numdata
            for k in result:
                evalvals[k].append(result[k]*numdata)
        if validation:
            validation_summary = tf.Summary()
        for k in evalvals:
            evalvals[k] = sum(evalvals[k])/total_numdata
            if validation:
                validation_summary.value.add(tag='validation/'+k, simple_value=evalvals[k])
        if validation:
            evalvals['summary'] = validation_summary 
        return evalvals
        
    def trainEpoch(self, bg, valbg, epoch, current_step_inside_epoch):
        if current_step_inside_epoch>0:
            num_step_left = bg.num_batches-current_step_inside_epoch-1
        train_accuracy = [] 
        train_loss = []
        for idx, item in enumerate(bg):
            if current_step_inside_epoch>0 and idx==num_step_left: break
            try:
                result = self.trainStep(item)
                train_accuracy.append(result['accuracy'])
                train_loss.append(result['loss'])
                if result['global_step'] % self.config.validfreq == 0:
                    print('Train: epoch {} step {}, loss {}, accuracy {}'.format(epoch, result['global_step'],
                    sum(train_loss)/len(train_loss), sum(train_accuracy)/len(train_accuracy)))
                    train_loss = []
                    valid_result = self.testStep(valbg, validation=True)
                    print('Validation: epoch {} step {}, loss {}, accuracy {}'.format(epoch, result['global_step'],
                    valid_result['loss'], valid_result['accuracy']))
                    self.summary_writer.add_summary(result['summary'], global_step=result['global_step'])
                if result['global_step'] % self.config.savefreq == 0 and valid_result['accuracy'] > result['max_val_accuracy']:
                    self.summary_writer.add_summary(valid_result['summary'], global_step=result['global_step'])
                    tf.assign(self.max_validation_accuracy, valid_result['accuracy'])
                    self.saveModel()
            except KeyboardInterrupt:
                self.sess.run(tf.assign(self.global_step, self.global_step-1))
                self.saveModel()
                sys.exit()
            except StopIteration:
                pass

    def saveModel(self):
        step = self.sess.run(self.global_step)
        print('\nSave {} model at iteration {}'.format(self.__class__.__name__, step))
        self.saver.save(self.sess, os.path.join(self.ckptpath, self.__class__.__name__), global_step=self.global_step) 

    def restoreModel(self, iternum=None):
        ckptstate = tf.train.get_checkpoint_state(self.ckptpath)
        if iternum: 
            file4restore = re.sub(r'\d+$', str(checkpoint_number), ckptstate.all_model_checkpoint_paths[-1])
        else:
            file4restore = ckptstate.model_checkpoint_path
        self.saver.restore(self.sess, file4restore)

    def train(self, trainBG, validBG, numepoch=10):
        current_step = self.sess.run(self.global_step)
        current_epoch = int(current_step/trainBG.num_batches)
        current_step_inside_epoch = current_step%trainBG.num_batches
        for epoch in range(current_epoch, numepoch):
            self.trainEpoch(trainBG, validBG, epoch, current_step_inside_epoch) 
            current_step_inside_epoch = 0
