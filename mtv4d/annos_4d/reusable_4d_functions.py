import sys
from xml.dom.expatbuilder import theDOMImplementation
sys.path.append(".")
from collections import defaultdict
from mtv4d.utils.io_base import read_json
import numpy as np
import os.path as op
from mtv4d.utils.misc_base import find_path_from_ts_and_dir_path_maybe_not_find
from pathlib import Path as P

# 这个脚本不被这个文件夹中其他内容引用！
def load_and_transfer_dn_boxes_from_frame(manual_frames_anno_dir, timestamps):
    dn_boxes_dict_ts2id, dn_boxes_dict_id2ts = defaultdict(dict), defaultdict(dict)
    if len(sorted(P(manual_frames_anno_dir).glob('*'))) == 0:
        return dn_boxes_dict_ts2id, dn_boxes_dict_id2ts
    for ts in timestamps:
        p = find_path_from_ts_and_dir_path_maybe_not_find(ts, manual_frames_anno_dir)
        # 主要时间花在这个find上面，可以考虑从timestamp直接获取文件名字。
        if p is not None:
            data = read_json(str(p))
            for box in data:
                box["ts"] = ts
                dn_boxes_dict_ts2id[ts][box["obj_track_id"]] = box
                dn_boxes_dict_id2ts[box["obj_track_id"]][ts] = box
    return dn_boxes_dict_ts2id, dn_boxes_dict_id2ts


def load_and_transfer_annos_from_map(manual_map_anno_dir):
    box_static_dict_id2info = {}
    map_boxes = read_json(manual_map_anno_dir)
    for i in map_boxes:
        box_static_dict_id2info[i["obj_track_id"]] = i
    return box_static_dict_id2info


def filter_ground_pts(scene_root, acc_pcl, ts, thres_dist=0.15):
    """
    acc_pcl是lidar系, ground_plane_model也是lidar系
    """
    file_name = find_path_from_ts_and_dir_path_maybe_not_find(ts, op.join(scene_root, "ground_plane_model"))
    if file_name is None:
        return acc_pcl[acc_pcl[:, 2] > -1.75, :3]
    abcd = np.loadtxt(file_name).reshape(4, 1)
    abcd = abcd * (2 * (abcd[2, 0] >= 0) - 1)
    acc_pcl_4d = np.concatenate([acc_pcl, np.ones([len(acc_pcl), 1])], axis=1)
    dist = acc_pcl_4d @ abcd
    acc_pcl = acc_pcl[dist.reshape(-1) > thres_dist]
    return acc_pcl


def find_near_points_index_of_visible_points(points, visible_inds, thres_near_point=0.1):
    from scipy.spatial.distance import cdist

    distances = cdist(points, points, "euclidean")
    distance_filtered = distances[visible_inds].reshape(len(visible_inds), len(points))
    idx = np.argsort(distance_filtered, axis=1)
    dist = np.take_along_axis(distance_filtered, idx, axis=1)
    return np.array(sorted(set(idx[dist < thres_near_point]))).astype("long")

def index_has_near_point_vis(points, visible_points, thres_near_point=0.1):
    from scipy.spatial.distance import cdist

    distances = cdist(points, visible_points, "euclidean")
    return np.arange(len(points))[distances.min(axis=1) < thres_near_point]