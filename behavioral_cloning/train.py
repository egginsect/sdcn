import cv2
import csv
import os
import ipdb
import random
from collections import defaultdict, OrderedDict
from batchgen import BatchGenerator
from model import BaseModel, LeNet
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import argparse
import sys
parser = argparse.ArgumentParser(description='Configs for training')
parser.add_argument('--split_threshold', default=.8, help='Training data percentage', type=float)
parser.add_argument('--gpu_fraction', default=.4, help='Percentage of GPU memory allowed to use', type=float)
parser.add_argument('--batch_size', default=30, help='Batch Size', type=int)
parser.add_argument('--num_epochs', default=5, help='Number of Epochs', type=int)
parser.add_argument('-i','--datadir', default='', help='input data', type=str)
parser.add_argument('-l', '--load_pretrained', dest='load_pretrained',help='Load pretrained model', action='store_true')
parser.set_defaults(load_pretrained=False)
config = parser.parse_args()
def get_session(gpu_fraction=config.gpu_fraction):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

data = [] 
imgDataName = ['center', 'left', 'right']
controllDataName = ['steering', 'throttle', 'brake', 'speed']
with open(os.path.join(config.datadir, 'driving_log.csv')) as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=imgDataName+controllDataName)
    for line in reader:
        data.append(line)
train_valid_split_idx = int(len(data)*config.split_threshold)
random.Random(29944).shuffle(data)
train_data, valid_data = data[:train_valid_split_idx], data[train_valid_split_idx:]
process_dict = OrderedDict()
process_dict['images'] = {'collect':['center'],
            'process':lambda fn: cv2.cvtColor(cv2.imread(os.path.join(config.datadir, 'IMG', os.path.basename(fn))), cv2.COLOR_BGR2RGB)} 
process_dict['controll']={'collect':['steering'],
            'process':float}

trainBG = BatchGenerator(train_data, process_dict, config.batch_size, phase='train')
validBG = BatchGenerator(valid_data, process_dict, config.batch_size, phase='test')
if config.load_pretrained:
    model = load_model('model.h5')
else:
    model = LeNet()
try:
    model.fit_generator(trainBG, steps_per_epoch=trainBG.num_batches, validation_data=validBG,
    validation_steps=validBG.num_batches, epochs=config.num_epochs)
except:
    model.save('model.h5')
    print('\n Model saved to model.h5')
    sys.exit()
model.save('model.h5')
