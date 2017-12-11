# Load pickled data
import pickle
import ipdb
import csv
training_file = 'traffic-signs-data/train.p'
validation_file= 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(y_train)

n_validation = len(y_valid)

n_test = len(y_test)

image_shape = X_train[0].shape

n_classes = len(set(y_train.tolist()))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import argparse
import cv2
from utils import BatchGenerator
from model import LeNetExt2
import argparse
parser = argparse.ArgumentParser(description='Configs for training')
parser.add_argument('--logdir', default='trainlog', help='Folder to save training logs', type=str)
parser.add_argument('--ckpt_number', help='Checkpoint number to be restored', type=int)
#parser.add_argument('-e', '--exprdir', default='trainlog/LeNet-2017-11-30-22:55-Adam-bs80-lr1.0e-03', help='Experiment case', type=str)
parser.add_argument('-e', '--exprdir', help='Experiment case', type=str)
parser.add_argument('--savefreq', default=1000, help='Batch Size', type=int)
parser.add_argument('--validfreq', default=100, help='Batch Size', type=int)
parser.add_argument('--optimizer', default='Adam', help='Optimizer used for the model', type=str)
parser.add_argument('--num_epoch', default=400, help='Number of epochs', type=int)
parser.add_argument('--learning_rate', default=1e-4, help='Learning rate', type=int)
parser.add_argument('--adaptive_learning_rate', default=False, help='Learning rate decay', type=bool)
parser.add_argument('--learning_rate_decay', default=0.01, help='Learning rate decay', type=float)
parser.add_argument('--batch_size', default=80, help='Batch Size', type=int)
parser.add_argument('--data_augmentation', default=True, help='Using data augmentation', type=int)
parser.add_argument('--grayscale', default=True, help='Using data augmentation', type=int)
parser.add_argument('--keep_prob', default=0.8, help='Keep probability for dropout', type=int)
config = parser.parse_args()
trainBG = BatchGenerator(X_train, y_train, config.batch_size, config.grayscale, 'train')
validBG = BatchGenerator(X_valid, y_valid, config.batch_size, config.grayscale)
testBG = BatchGenerator(X_valid, y_valid, config.batch_size, config.grayscale)
config.decay_steps = trainBG.num_batches*config.num_epoch
label_dict =  {}
with open('signnames.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label_dict[row['ClassId']] = row['SignName']
#vgg = VGGsimple(config, label_dict)
#vgg.train(trainBG, validBG, config.num_epoch)
lenet = LeNetExt2(config, label_dict)
lenet.train(trainBG, validBG, config.num_epoch)
