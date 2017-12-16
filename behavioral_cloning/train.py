import cv2
import csv
import os
import ipdb
from collections import defaultdict, OrderedDict
from batchgen import BatchGenerator
from model import BaseModel, LeNet
from keras.models import load_model
import argparse
import sys
parser = argparse.ArgumentParser(description='Configs for training')
parser.add_argument('--split_threshod', default=.8, help='Training data percentage', type=float)
parser.add_argument('--batch_size', default=20, help='Batch Size', type=int)
parser.add_argument('-i','--datadir', default='', help='input data', type=str)
parser.add_argument('-l', '--load_pretrained', dest='load_pretrained',help='Load pretrained model', action='store_true')
parser.set_defaults(load_pretrained=False)
config = parser.parse_args()
data = [] 
imgDataName = ['center', 'left', 'right']
controllDataName = ['steering', 'throttle', 'brake', 'speed']
with open(os.path.join(config.datadir, 'driving_log.csv')) as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=imgDataName+controllDataName)
    for line in reader:
        data.append(line)
train_valid_split_idx = int(len(data)*config.split_threshod)
train_data, valid_data = data[:train_valid_split_idx], data[train_valid_split_idx:]
process_dict = OrderedDict()
process_dict['images'] = {'collect':['center'],
            'process':lambda fn: cv2.imread(os.path.join(config.datadir, 'IMG', os.path.basename(fn)))} 
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
    validation_steps=validBG.num_batches, epochs=40)
except KeyboardInterrupt:
    model.save('model.h5')
    print('\n Model saved to model.h5')
    sys.exit()
model.save('model.h5')
