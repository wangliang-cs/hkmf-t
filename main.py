# Copyright (c) [2021] [wlicsnju]
# [HKMF-T] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2. 
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2 
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
# See the Mulan PSL v2 for more details.  
import datetime
import fire
import logging

import numpy as np
import matplotlib.pyplot as plt

from dtw import dtw
from typing import Union, Callable
from joblib import Parallel, delayed, cpu_count

import utils

from callback import EpochConvergeCallback, MaxDiffConvergeCallback
from hkmft import HKMFT, HKMFTTrainParam, HKMFT_MAX_EPOCH
from ma_tag import MATag
from tag_mean import TagMean
from linear import LinearInterpolation


def rmse_metric(gt: np.ndarray, rs: np.ndarray):
    return np.sqrt(np.average((gt - rs) ** 2))


def my_dtw_metric(gt: np.ndarray, rs: np.ndarray):
    if gt.shape != rs.shape:
        logging.error(f'Ground truth shape {gt.shape} do not match to result shape {rs.shape}!', ValueError)
        return
    if len(gt.shape) == 1:
        gt = gt[np.newaxis, :]
        rs = rs[np.newaxis, :]
    elif len(gt.shape) == 2:
        pass
    else:
        logging.error(f'Ground truth {gt.shape} must 1-dim or 2-dim!', ValueError)
        return
    dist = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
    DTW = np.zeros((gt.shape[-1], rs.shape[-1]))
    DTW[0, 0] = dist(gt[:, 0], rs[:, 0])
    for i in range(1, gt.shape[-1]):
        DTW[i, 0] = DTW[i - 1, 0] + dist(gt[:, i], rs[:, 0])
    for j in range(1, rs.shape[-1]):
        DTW[0, j] = DTW[0, j - 1] + dist(gt[:, 0], rs[:, j])
    for i in range(1, gt.shape[-1]):
        for j in range(1, rs.shape[-1]):
            DTW[i, j] = dist(gt[:, i], rs[:, j]) + min(DTW[i - 1, j],
                                                       DTW[i, j - 1],
                                                       DTW[i - 1, j - 1])
    return DTW[-1, -1] / gt.shape[-1]


def dtw_metric(gt: np.ndarray, rs: np.ndarray):
    if gt.shape != rs.shape:
        logging.error(f'Ground truth shape {gt.shape} do not match to result shape {rs.shape}!', ValueError)
        return
    if len(gt.shape) == 1:
        gt = gt[np.newaxis, :]
        rs = rs[np.newaxis, :]
    elif len(gt.shape) == 2:
        pass
    else:
        logging.error(f'Ground truth {gt.shape} must 1-dim or 2-dim!', ValueError)
        return
    dtw_v, *_ = dtw(gt.transpose(), rs.transpose(), dist=lambda x, y: np.sqrt(np.sum((x - y) ** 2)))
    return dtw_v / gt.shape[-1]


def get_method_handle(method: str) -> Union[Callable, None]:
    if method not in {'hkmft', 'tagmean', 'linear', 'matag'}:
        return None
    if method == 'hkmft':
        return hkmft_core
    elif method == 'tagmean':
        return tagmean_core
    elif method == 'linear':
        return linear_core
    elif method == 'matag':
        return matag_core


def hkmft_core(data, mask, tag, gt,
               max_epoch: int = HKMFT_MAX_EPOCH,
               train_eta: float = 0.01,
               train_lambda_s: float = 0.1,
               train_lambda_o: float = 0.001,
               train_lambda_e: float = 0.1,
               train_stop_rate: float = 1.0,
               train_converge_threshold: float = 0.001,
               *args,
               ):
    callbacks = [EpochConvergeCallback(max_epoch),
                 MaxDiffConvergeCallback(train_converge_threshold)]
    m = HKMFT()
    m.put_and_reset(data, mask, tag)
    m.train(HKMFTTrainParam(eta=train_eta, lambda_s=train_lambda_s,
                            lambda_o=train_lambda_o, lambda_e=train_lambda_e,
                            stop_rate=train_stop_rate, random=False), callbacks)
    rs = m.get_result()
    rmse_score = rmse_metric(gt, rs)
    dtw_score = dtw_metric(gt, rs)
    return rs, rmse_score, dtw_score


def tagmean_core(data, mask, tag, gt, *args):
    m = TagMean()
    m.put_and_reset(data, mask, tag)
    rs = m.get_result()
    rmse_score = rmse_metric(gt, rs)
    dtw_score = dtw_metric(gt, rs)
    return rs, rmse_score, dtw_score


def linear_core(data, mask, tag, gt, *args):
    m = LinearInterpolation()
    m.put_and_reset(data, mask, tag)
    rs = m.get_result()
    rmse_score = rmse_metric(gt, rs)
    dtw_score = dtw_metric(gt, rs)
    return rs, rmse_score, dtw_score


def matag_core(data, mask, tag, gt, *args):
    m = MATag()
    m.put_and_reset(data, mask, tag)
    rs = m.get_result()
    rmse_score = rmse_metric(gt, rs)
    dtw_score = dtw_metric(gt, rs)
    return rs, rmse_score, dtw_score


def show_main(*result_files,
              dataset: str = None,
              blackouts_begin: int = None,
              blackouts_len: int = None,
              no_rmse: bool = False,
              no_dtw: bool = False,
              recalculate: bool = False,
              ):
    """
    :param result_files: result files saved in plk format. If flags --dataset, --blackouts_begin, and --blackouts_len are set, the program shows the detailed recovering results for a single period. Otherwise, it shows an overview of the recovering errors (distances) with respect to the length of blackouts.
    :param dataset: must in ['BSD', 'MVCD', 'EPCD'].
    :param blackouts_begin: blackouts begin index, closed.
    :param blackouts_len: blackouts length, must int.
    :param no_rmse: do not show rmse.
    :param no_dtw: do not show dtw.
    :param recalculate: recalculate metrics then show it.
    :return:
    """
    logging.basicConfig(format='%(asctime)-15s %(message)s',
                        level=logging.INFO)
    # return dict like {'dataset_name': {blackout_lens: (start_idx, results, params)}}
    results = utils.results_load(result_files)
    # single plot
    if dataset is not None and blackouts_begin is not None and blackouts_len is not None:
        dl = utils.dataset_load(dataset)
        if dl is None:
            return -1
        if blackouts_len not in results[dataset]:
            logging.error(f'blackouts_len {blackouts_len} do not exists in given files!', ValueError)
            return -1
        rs = results[dataset][blackouts_len]
        if blackouts_begin not in rs[0]:
            logging.error(f'blackouts_begin {blackouts_begin} do not exists in given files!', ValueError)
            return -1
        dl.norm(0.0, rs[2]['dataset_norm'])
        dl.generate_mask(blackouts_begin, blackouts_begin + blackouts_len)
        _, _, tag, gt = dl.get_data()
        idx = rs[0].index(blackouts_begin)
        utils.show_gt_rs(gt, rs[1][idx][0], blackouts_begin, blackouts_begin + blackouts_len, rs[2]['method'])
        return 0
    # outlook plot
    ground_truth = {}
    if recalculate:
        for ds in results:
            dl = utils.dataset_load(ds)
            dl.norm()  # results 采用不同参数时, 此处可能会出现问题.
            ground_truth[ds], *_ = dl.get_data()
    ds_keys = {}
    for ds in results:
        lens = []
        for l in results[ds]:
            lens.append(l)
        ds_keys[ds] = lens
    if len(ds_keys) <= 0:
        return -1
    fig, axs = plt.subplots(len(ds_keys), 1)
    if len(ds_keys) == 1:
        axs = [axs]
    for i, ds in enumerate(ds_keys):
        rmse_avg, dtw_avg = [], []
        for l in ds_keys[ds]:
            if recalculate:
                rmse_result = [rmse_metric(ground_truth[ds][..., results[ds][l][0][idx]:
                                                            results[ds][l][0][idx] + l], r[0])
                               for idx, r in enumerate(results[ds][l][1])]
                dtw_result = [dtw_metric(ground_truth[ds][..., results[ds][l][0][idx]:
                                                          results[ds][l][0][idx] + l], r[0])
                              for idx, r in enumerate(results[ds][l][1])]
            else:
                rmse_result = [r[1] for r in results[ds][l][1]]
                dtw_result = [r[2] for r in results[ds][l][1]]
            rmse_avg.append(np.average(rmse_result))
            dtw_avg.append(np.average(dtw_result))
        if not no_rmse:
            print(f'{ds} RMSE: {rmse_avg}\n')
            axs[i].plot(ds_keys[ds], rmse_avg, 'o-', label=f'{ds}_rmse')
        if not no_dtw:
            print(f'{ds} DTW: {dtw_avg}\n')
            axs[i].plot(ds_keys[ds], dtw_avg, '^-', label=f'{ds}_dtw')
        axs[i].set_title(f'{ds} results')
        axs[i].set_ylabel('distance')
        axs[i].legend()
    axs[-1].set_xlabel('blackouts length')
    plt.show()


def enum_main(dataset: str,
              blackouts_lens: Union[int, str],
              step_len: int = None,
              dataset_norm: float = 10.0,
              method: str = 'hkmft',
              max_epoch: int = HKMFT_MAX_EPOCH,
              train_eta: float = 0.01,
              train_lambda_s: float = 0.1,
              train_lambda_o: float = 0.001,
              train_lambda_e: float = 0.1,
              train_stop_rate: float = 1.0,
              train_converge_threshold: float = 0.001
              ):
    """
    HKMF-T enum mode.
    :param dataset: must in ['BSD', 'MVCD', 'EPCD'].
    :param blackouts_lens: the interval [x, y] of blackout lengths, given in the form of x-y, and x <= y.
    :param step_len: blackouts begin step length. (default 1)
    :param dataset_norm: data 0-dataset_norm normalize. (default 10.0)
    :param method: must in ['hkmft', 'tagmean', 'linear', 'matag']. (default 'hkmft')
    :param max_epoch: (hkmft) max epoch, if converge will be exit early. (default HKMFT_MAX_EPOCH=5000)
    :param train_eta: (hkmft) train parameter, η. (default 0.01)
    :param train_lambda_s: (hkmft) train parameter, λ_s. (default 0.1)
    :param train_lambda_o: (hkmft) train parameter, λ_o. (default 0.001)
    :param train_lambda_e: (hkmft) train parameter, λ_e. (default 0.1)
    :param train_stop_rate: (hkmft) train parameter, s. (default 1.0)
    :param train_converge_threshold: (hkmft) converge if diff less than threshold. (default 0.001)
    :return:
    """
    logging.basicConfig(format='%(asctime)-15s %(message)s',
                        level=logging.ERROR)
    params = {
        'dataset': dataset,
        'blackouts_lens': blackouts_lens,
        'dataset_norm': dataset_norm,
        'method': method,
        'max_epoch': max_epoch,
        'train_eta': train_eta,
        'train_lambda_s': train_lambda_s,
        'train_lambda_o': train_lambda_o,
        'train_lambda_e': train_lambda_e,
        'train_stop_rate': train_stop_rate,
        'train_converge_threshold': train_converge_threshold,
    }
    dl = utils.dataset_load(dataset)
    if dl is None:
        return -1
    method_core = get_method_handle(method)
    if method_core is None:
        logging.error(f'method {method} is does not supported yet.')
        return -1
    if dataset_norm < 1e-3:
        logging.error(f'dataset_norm {dataset_norm} must > 0.0!', ValueError)
        return -1
    if max_epoch < 0 or train_eta < 0 or train_lambda_s < 0 or train_lambda_o < 0 or train_lambda_e < 0:
        logging.error(f'Train params must > 0!', ValueError)
        return -1
    if step_len is not None:
        if step_len < 1:
            logging.error(f'Param step_len must >= 1!', ValueError)
            return -1
        step_len = int(step_len)

    dl.norm(0.0, dataset_norm)
    blackouts_lens = utils.lens_to_list(blackouts_lens)
    if blackouts_lens is None:
        return -1

    start_idx = []

    def _generator():
        for l in blackouts_lens:
            if step_len is None:
                sl = l
            else:
                sl = step_len
            for i in range(0, len(dl) - l - l, sl):
                dl.generate_mask(l + i, l + l + i)
                start_idx.append(l + i)
                yield dl.get_data()

    results = Parallel(n_jobs=(cpu_count() + 2) // 3, prefer='processes', verbose=1)(
        delayed(method_core)(data, mask, tag, gt,
                             max_epoch,
                             train_eta,
                             train_lambda_s,
                             train_lambda_o,
                             train_lambda_e,
                             train_stop_rate,
                             train_converge_threshold, )
        for data, mask, tag, gt in _generator()
    )
    utils.result_save(f'results_{dataset}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.plk',
                      params, start_idx, results)


def single_main(dataset: str,
                blackouts_begin: int,
                blackouts_end: int,
                dataset_norm: float = 10.0,
                method: str = 'hkmft',
                max_epoch: int = HKMFT_MAX_EPOCH,
                train_eta: float = 0.01,
                train_lambda_s: float = 0.1,
                train_lambda_o: float = 0.001,
                train_lambda_e: float = 0.1,
                train_stop_rate: float = 1.0,
                train_converge_threshold: float = 0.001,
                is_show: bool = True
                ):
    """
    HKMF-T single mode.
    :param dataset: must in ['BSD', 'MVCD', 'EPCD'].
    :param blackouts_begin: blackouts begin index, closed.
    :param blackouts_end: blackouts end index, open. [blackouts_begin, blackouts_end)
    :param dataset_norm: data 0-dataset_norm normalize. (default 10.0)
    :param method: must in ['hkmft', 'tagmean', 'linear', 'matag']. (default 'hkmft')
    :param max_epoch: (hkmft) max epoch, if converge will be exit early. (default HKMFT_MAX_EPOCH=5000)
    :param train_eta: (hkmft) train parameter, η. (default 0.01)
    :param train_lambda_s: (hkmft) train parameter, λ_s. (default 0.1)
    :param train_lambda_o: (hkmft) train parameter, λ_o. (default 0.001)
    :param train_lambda_e: (hkmft) train parameter, λ_e. (default 0.1)
    :param train_stop_rate: (hkmft) train parameter, s. (default 1.0)
    :param train_converge_threshold: (hkmft) converge if diff less than threshold. (default 0.001)
    :param is_show: show result and ground truth in graphical. (default True)
    :return:
    """
    logging.basicConfig(format='%(asctime)-15s %(message)s',
                        level=logging.INFO)
    # param valid
    dl = utils.dataset_load(dataset)
    if dl is None:
        return -1
    method_core = get_method_handle(method)
    if method_core is None:
        logging.error(f'method {method} is does not supported yet.')
        return -1
    if dataset_norm < 1e-3:
        logging.error(f'dataset_norm {dataset_norm} must > 0.0!', ValueError)
        return -1
    if blackouts_end == blackouts_begin:
        return 0
    if blackouts_end < blackouts_begin:
        blackouts_begin, blackouts_end = blackouts_end, blackouts_begin
    if blackouts_begin < (blackouts_end - blackouts_begin) or blackouts_end > len(dl):
        logging.error(f'blackouts_start/blackouts_end error!', ValueError)
        return -1
    if max_epoch < 0 or train_eta < 0 or train_lambda_s < 0 or train_lambda_o < 0 or train_lambda_e < 0:
        logging.error(f'Train params must > 0!', ValueError)
        return -1

    # main process
    dl.norm(0.0, dataset_norm)
    dl.generate_mask(blackouts_begin, blackouts_end)
    data, mask, tag, gt = dl.get_data()
    rs, rmse_score, dtw_score = method_core(data, mask, tag, gt,
                                            max_epoch,
                                            train_eta,
                                            train_lambda_s,
                                            train_lambda_o,
                                            train_lambda_e,
                                            train_stop_rate,
                                            train_converge_threshold, )
    print('gt:', gt)
    print('rs:', rs)
    print('rmse_score:', rmse_score)
    print('dtw_score:', dtw_score)
    print('dtw_score:', dtw_metric(gt, rs))
    if is_show:
        utils.show_gt_rs(gt, rs, blackouts_begin, blackouts_end, method)
    return 0


if __name__ == '__main__':
    """
    single --dataset BSD --blackouts_begin 289 --blackouts_end 295 --max_epoch 200
    show .\results_BSD_20210612_055352.plk
    """
    fire.Fire({'enum': enum_main,
               'single': single_main,
               'show': show_main})
