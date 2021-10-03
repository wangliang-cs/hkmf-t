import logging

import numpy as np


def find_max_continue_0(m: np.ndarray):
    if len(m.shape) == 1:
        tmp = [m]
    elif len(m.shape) == 2:
        tmp = m
    else:
        logging.error('Do not support mask with dim > 2.', ValueError)
        return 0
    max_con0 = 0
    for t in tmp:
        count = 0
        for x in t:
            if x != 0:
                max_con0 = max(max_con0, count)
                count = 0
            else:
                count += 1
    return max_con0


def hankelization(data, mask, tag, p):
    if data.shape != tag.shape or len(mask.shape) != 1:
        return
    if len(data.shape) == 1:
        data = data[np.newaxis, :]
        tag = tag[np.newaxis, :]
    elif len(data.shape) == 2:
        pass
    elif len(data.shape) > 2:
        return
    data_d = data.shape[0]
    data_l = data.shape[1]
    Hpx = np.zeros((data_d * p, data_l))
    Hpx_mask = np.zeros((data_d * p, data_l), dtype=np.int8)
    Hpx_tag = np.full((data_d * p, data_l), -1, dtype=np.int)
    for i in range(len(data)):
        for j in range(p):
            Hpx[i * p + j, :(data_l - j)] = data[i, j:]
            Hpx_mask[i * p + j, :(data_l - j)] = mask[j:]
            Hpx_tag[i * p + j, :(data_l - j)] = tag[i, j:]
    return Hpx, Hpx_mask, Hpx_tag


def get_hankel_result(A: np.ndarray, mask: np.ndarray, p: int) -> np.ndarray:
    blackout_l = np.sum(mask[0, :] == 0)
    data_d = mask.shape[0] // p
    data_l = mask.shape[1]
    rs = np.zeros((data_d, blackout_l), dtype=float)
    for i in range(data_d):
        for j in range(p):
            c = 0
            for k, x in enumerate(mask[i * p + j, :(data_l - j)]):
                if x == 0:
                    rs[i, c] += A[i * p + j, k]
                    c += 1
        rs[i, :] /= float(p)
    return rs
