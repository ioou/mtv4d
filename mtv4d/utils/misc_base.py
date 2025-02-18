import os
from time import time
from collections import defaultdict
import torch.multiprocessing as torch_mp
import multiprocessing as mp
from tqdm import tqdm
import os.path as op
import numpy as np


class Time:
    def __init__(self):
        self.time = time()

    def tic(self):
        self.time = time()

    def toc(self, info=None):
        if info is not None:
            print(time() - self.time, info)
        else:
            print(time() - self.time)
        self.time = time()


def defaultdict_lambda():
    return defaultdict(dict)


def torch_pool(func, list_to_process, n_processes=torch_mp.cpu_count()):
    torch_mp.set_start_method("spawn", force=True)
    with torch_mp.Pool(n_processes) as pool:
        r = list(tqdm(pool.imap(func, list_to_process), total=len(list_to_process)))
    return r


def mp_pool(func, list_to_process, n_processes=mp.cpu_count()):
    with mp.Pool(n_processes) as p:
        r = list(tqdm(p.imap(func, list_to_process), total=len(list_to_process)))
    return r


def get_sync_filename(file_names, file_times, sync_time, return_idx=False):
    """
    get sync filename in file_root according to the sync_time
    """
    times_diff = np.abs(file_times - sync_time)
    min_index = np.argmin(times_diff)
    if times_diff[min_index] < 15:
        sync_file_name = file_names[min_index]
        if return_idx:
            return sync_file_name, min_index
        return sync_file_name
    else:
        if return_idx:
            return None, None
        return None


def get_times(file_root, sync_offset=None, no_sync_list=None, return_frame=False, tag=None):
    if sync_offset is not None:
        id = op.basename(file_root)
    file_names = os.listdir(file_root)
    times = []
    file_names = [x for x in file_names if not "html" in x]
    if tag is not None:
        file_names = [x for x in file_names if tag in x]
    frames = []
    for file_name in file_names:
        if "html" in file_name:
            time = 0
        elif "indices" in file_name:
            time = 0
        elif "_" in file_name:
            time = float(file_name.split("_")[1].split(".")[0])
            frames.append(int(file_name.split("_")[0]))
        else:
            time = float(file_name.split(".")[0])
        if sync_offset is not None:
            assert no_sync_list is not None, "must input no_sync_list"
            if not id in no_sync_list:
                time -= sync_offset
        times.append(time)
    times = np.array(times)
    file_names = np.array(file_names)
    if len(frames) > 0:
        frames = np.array(frames)
        sort_ind = np.argsort(frames)
        frames = frames[sort_ind]
    else:
        sort_ind = np.argsort(times)
    times = times[sort_ind]
    file_names = file_names[sort_ind]
    if return_frame and len(frames) > 0:
        return times, file_names, frames
    else:
        return times, file_names


def get_round_timestamp(timestamps, filter_unsync=False):
    round_timestamps = (np.round(timestamps / 100) * 100).astype(np.int64)
    offsets = abs(round_timestamps - timestamps)
    valid_mask = offsets < 15
    valid_round_timestamps = round_timestamps[valid_mask]
    if filter_unsync:
        return valid_round_timestamps, valid_mask
    else:
        return round_timestamps


def find_path_from_ts_and_dir_path(ts, dir_path):
    times, file_names = get_times(dir_path)
    return op.join(dir_path, get_sync_filename(file_names, times, ts))


def find_path_from_ts_and_dir_path_maybe_not_find(ts, dir_path):
    times, file_names = get_times(dir_path)
    if get_sync_filename(file_names, times, ts) is None:
        return None
    return op.join(dir_path, get_sync_filename(file_names, times, ts))
