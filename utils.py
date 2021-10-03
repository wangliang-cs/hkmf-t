import logging
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from typing import Union

from dataloder import DataLoader
from dataloder import Dataset


def show_gt_rs(gt: np.ndarray, rs: np.ndarray, begin_idx, end_idx, method_name: str = 'HKMF-T') -> None:
    if gt.shape != rs.shape:
        logging.error(f'Ground truth shape {gt.shape} do not match to result shape {rs.shape}!', ValueError)
        return
    if len(gt.shape) == 2 and gt.shape[0] == 1:
        gt = gt[0, :]
        rs = rs[0, :]
    if len(gt.shape) == 1:
        plt.plot(range(begin_idx, end_idx), gt, label='ground truth')
        plt.plot(range(begin_idx, end_idx), rs, label=method_name)
        plt.legend()
        plt.show()
    elif len(gt.shape) == 2:
        for i in range(gt.shape[0]):
            plt.plot(range(begin_idx, end_idx), gt[i], label=f'ground truth[{i}]')
            plt.plot(range(begin_idx, end_idx), rs[i], label=f'{method_name}[{i}]')
        plt.xticks(range(begin_idx, end_idx))
        plt.legend()
        plt.show()
    else:
        logging.error(f'Ground truth {gt.shape} must 1-dim or 2-dim!', ValueError)
        return


def lens_to_list(lens: Union[int, str]) -> Union[list, None]:
    if isinstance(lens, int):
        lens_list = [lens, ]
    elif isinstance(lens, str):
        _ = lens.split('-')
        if len(_) != 2:
            logging.error(f'blackouts_lens {lens} must int or str(int-int)!', ValueError)
            return None
        lens_list = range(int(_[0]), int(_[1]) + 1)
    else:
        logging.error(f'blackouts_lens {lens} must int or str(int-int)!', ValueError)
        return None
    return lens_list


def dataset_load(dataset: str) -> Union[DataLoader, None]:
    ds = None
    for d in Dataset:
        if dataset == d.name:
            ds = d
            break
    if ds is None:
        logging.error(f'dataset {dataset} do not exist!', ValueError)
        return None
    return DataLoader(ds)


def result_save(filename: str, params: dict, start_idx: list, results: list) -> None:
    with open(filename, 'wb') as fp:
        pickle.dump({
            'params': params,
            'start_idx': start_idx,
            'results': results,
        }, fp)
        fp.close()


def _find_fall(li: list, s: int) -> int:
    r = s
    for i in range(s + 1, len(li)):
        if li[i] < li[r]:
            return i
        r += 1


def results_load(filenames: Union[tuple, list, str]) -> dict:
    """
    load result files, return dict like {'dataset_name': {blackout_lens: (start_idx, results, params)}}
    :param filenames: list of filenames.
    :return: dict.
    """
    if isinstance(filenames, str):
        filenames = (filenames, )
    results = {}
    for fn in filenames:
        if os.path.exists(fn):
            with open(fn, 'rb') as fp:
                try:
                    obj = pickle.load(fp)
                except IOError:
                    logging.warning(f'Input file {fn} is not a pickle file.')
                else:
                    if (isinstance(obj, dict) and
                            ('params' in obj) and
                            ('start_idx' in obj) and
                            ('results' in obj)):
                        ds_name = obj['params']['dataset']
                        lens = lens_to_list(obj['params']['blackouts_lens'])
                        if ds_name not in results:
                            results[ds_name] = {}
                        l = 0
                        for i in lens:
                            r = _find_fall(obj['start_idx'], l)
                            if i not in results[ds_name]:
                                params = dict(obj['params'])
                                params['blackouts_lens'] = i
                                results[ds_name][i] = (obj['start_idx'][l:r],
                                                       obj['results'][l:r],
                                                       params)
                            else:
                                logging.warning(f'Input file {fn} {ds_name} {i} repeated, ignore.')
                            l = r
                    else:
                        logging.warning(f'Input file {fn} is not a hkmf results file.')
        else:
            logging.warning(f'Input file {fn} do not exists.')
    return results
