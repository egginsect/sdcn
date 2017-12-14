from random import shuffle
from math import ceil
import cv2
import numpy as np
from collections import defaultdict, OrderedDict
from itertools import cycle, chain
from random import choice
import numpy as np
class BatchGenerator(object):
    def __init__(self, data, process_dict, batch_size, phase='test'):
        self.batch_size = batch_size
        self.data = data 
        self.process_dict = process_dict
        self.phase = phase
        self.num_data = len(self.data)
        self.num_batches = ceil(float(len(self.data))/batch_size)
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
        return np.vstack(batchlist[0]), np.vstack(batchlist[1])

    def __iter__(self):
        self.reload()
        try:
            while(True):
                yield(self.__next__())
        except StopIteration:
            self.reload()

    def data_augmentation(self, processed):
        for item in processed['images']:
            yield [item], processed['controll']

    def preprocess(self, datum):
        processed = OrderedDict() 
        for key, value in self.process_dict.items():
            tmp = [datum[dataname]for dataname in value['collect']]
            processed[key] = list(map(value['process'], tmp))
        return self.data_augmentation(processed)

    def reload(self):
        shuffle(self.data)
        self.data_iter = cycle(chain.from_iterable(map(self.preprocess, self.data)))
