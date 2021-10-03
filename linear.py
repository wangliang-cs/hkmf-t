import logging

import numpy as np


class LinearInterpolation(object):
    def __init__(self):
        self._data = None
        self._mask = None
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
        elif len(data.shape) == 2:
            self._data = data
        elif len(data.shape) > 2:
            logging.error('Do not support data with dim > 2.', ValueError)
            return
        self._dim = self._data.shape[0]
        self._mask = mask

    def train(self, *args):
        logging.warning('Linear Interpolation class do not train for result.')

    def get_result(self):
        blackout_l = np.sum(self._mask == 0)
        rs = np.zeros((self._dim, blackout_l), dtype=float)

        idx = np.arange(self._data.shape[1])
        for d in range(self._dim):
            xp = idx[self._mask > 0]
            fp = self._data[d, self._mask > 0]
            est_x = idx[self._mask <= 0]
            if len(est_x) > 0:
                rs[d, :] = np.interp(est_x, xp, fp)
        return rs
