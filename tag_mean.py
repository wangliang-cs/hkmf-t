import logging

import numpy as np


class TagMean(object):
    def __init__(self):
        self._data = None
        self._mask = None
        self._tag = None
        self._dim = 1

    def put_and_reset(self, data, mask, tag):
        if data.shape != tag.shape:
            logging.error(f'\'data\'{data.shape} must be same to \'tag\'{tag.shape}.', ValueError)
            return
        if len(mask.shape) != 1:
            logging.error(f'\'mask\'{mask.shape} must be 1-d array.', ValueError)
            return
        if len(data.shape) == 1:
            self._data = data[np.newaxis, :]
            tag = tag[np.newaxis, :]
        elif len(data.shape) == 2:
            self._data = data
        elif len(data.shape) > 2:
            logging.error('Do not support data with dim > 2.', ValueError)
            return
        self._dim = self._data.shape[0]
        self._mask = mask
        self._tag = tag

    def train(self, *args):
        logging.warning('Tag Mean class do not train for result.')

    def get_result(self):
        all_avg = np.zeros((self._dim, ))
        tag_sum = {}
        tag_count = {}
        for i, m in enumerate(self._mask):
            tag = self._tag[0, i]
            if m > 0:
                if tag not in tag_sum:
                    tag_sum[tag] = np.zeros((self._dim, ))
                    tag_count[tag] = 0
                all_avg += self._data[:, i]
                tag_sum[tag] += self._data[:, i]
                tag_count[tag] += 1
        all_avg = all_avg / float(np.sum(self._mask != 0))

        blackout_l = np.sum(self._mask == 0)
        rs = np.zeros((self._dim, blackout_l), dtype=float)

        j = 0
        for i, m in enumerate(self._mask):
            tag = self._tag[0, i]
            if m == 0:
                if tag not in tag_sum:
                    rs[:, j] = all_avg
                else:
                    rs[:, j] = tag_sum[tag] / float(tag_count[tag])
                j += 1
        return rs
