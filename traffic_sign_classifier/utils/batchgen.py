from random import shuffle
from math import ceil
import cv2
import numpy as np
from collections import defaultdict
from random import choice
import ipdb
class BatchGenerator(object):
    def __init__(self, imgs, labels, batch_size, grayscale, phase='test'):
        self.batch_size = batch_size
        self.grayscale = grayscale
        self.phase = phase
        if phase=='test':
            self.pairs = list(zip(imgs, labels))
        else:
            self.pairs = self.balanceData(imgs, labels)
        self.num_data = len(self.pairs)
        self.num_batches = ceil(float(len(self.pairs))/batch_size)
        self.processeddata = None
        self.reload()
     
    def balanceData(self, data, labels):
        data_dict = defaultdict(list)
        for datum, label in zip(data, labels):
            data_dict[label].append(datum)
        max_len = max([len(value) for key,value in data_dict.items()])
        pairs = [] 
        for key, value in data_dict.items():
            pairs+=[(choice(value), key)for _ in range(max_len)]
        return pairs

    def __next__(self):
        batchlist = list(zip(*[next(self.data_iter) for _ in range(self.batch_size)]))
    
        return {'images':batchlist[0], 'labels':batchlist[1]}

    def __iter__(self):
        self.reload()
        try:
            while(True):
                yield(self.__next__())
        except StopIteration:
            pass

    def preprocess(self, pair):
        img = pair[0]
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:,None]
        return (img-128)/128 , pair[1]
    
    def reload(self):
        shuffle(self.pairs)
        self.data_iter = map(self.preprocess, self.pairs)

    def singleBatchWithAllData(self):
        self.reload()
        batchlist = list(zip(*[item for item in self.data_iter]))
        if not(self.processeddata):
            self.processeddata = {'images':batchlist[0], 'labels':batchlist[1]}  
        return self.processeddata 
