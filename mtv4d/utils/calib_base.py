import sys
sys.path.append('.')
import os.path as op
from easydict import EasyDict
import yaml
import numpy as np
from scipy.spatial.transform import Rotation

def format_cal_data(cal_data):
    """
    generate extrinsic mat, camera intrinsic mat and distort coefficient
    """
    sids = list(cal_data.keys())
    # if 'cam_bev' in sids:
    #     sids.remove('cam_bev')
    for sid in sids:
        if "cameras" in cal_data[sid]["sensor_model"]:
            if not "muzza" in cal_data[sid]["sensor_model"].lower():  # only support opencv model
                focal = cal_data[sid]["focal"]
                pp = cal_data[sid]["pp"]
                K = np.eye(3)
                if isinstance(focal, list):
                    K[0, 0], K[1, 1] = focal
                else:
                    K[0, 0], K[1, 1] = focal, focal
                K[0, 2], K[1, 2] = pp
                cal_data[sid]["K"] = K
                if "inv_poly" in cal_data[sid].keys():
                    cal_data[sid]["D"] = np.array(cal_data[sid]["inv_poly"])
        if "extrinsic" in cal_data[sid].keys():
            tx, ty, tz, qx, qy, qz, qw = cal_data[sid]["extrinsic"]
            T = np.eye(4)
            q = Rotation.from_quat([qx, qy, qz, qw])
            if hasattr(q, "as_dcm"):
                T[:3, :3] = q.as_dcm().astype("float32")
            else:
                T[:3, :3] = q.as_matrix().astype("float32")
            T[:3, 3] = np.array([tx, ty, tz])
            cal_data[sid]["T_es"] = T
            cal_data[sid]["T_se"] = np.linalg.inv(T)


def read_cal_data(yaml_file):
    cal_data = EasyDict()
    with open(yaml_file) as f:
        rig_data = yaml.load(f, Loader=yaml.FullLoader)
        rig_data = EasyDict(rig_data)
        # camera_cal_file_version = rig_data.get('version',1.0)
        cal_data = rig_data.rig
        format_cal_data(cal_data)
        # if camera_cal_file_version<=1.0:
        #     transform_old_rig(cal_data)
    return cal_data


def format_cal_data(cal_data):
    """
    generate extrinsic mat, camera intrinsic mat and distort coefficient
    """
    sids = list(cal_data.keys())
    for sid in sids:
        if "cameras" in cal_data[sid]["sensor_model"]:
            if not "muzza" in cal_data[sid]["sensor_model"].lower():  # only support opencv model
                focal = cal_data[sid]["focal"]
                pp = cal_data[sid]["pp"]
                K = np.eye(3)
                if isinstance(focal, list):
                    K[0, 0], K[1, 1] = focal
                else:
                    K[0, 0], K[1, 1] = focal, focal
                K[0, 2], K[1, 2] = pp
                cal_data[sid]["K"] = K
                if "inv_poly" in cal_data[sid].keys():
                    cal_data[sid]["D"] = np.array(cal_data[sid]["inv_poly"])
        if "extrinsic" in cal_data[sid].keys():
            tx, ty, tz, qx, qy, qz, qw = cal_data[sid]["extrinsic"]
            T = np.eye(4)
            q = Rotation.from_quat([qx, qy, qz, qw])
            if hasattr(q, "as_dcm"):
                T[:3, :3] = q.as_dcm().astype("float32")
            else:
                T[:3, :3] = q.as_matrix().astype("float32")
            T[:3, 3] = np.array([tx, ty, tz])
            cal_data[sid]["T_es"] = T
            cal_data[sid]["T_se"] = np.linalg.inv(T)

def read_ego_paths(trajectory_path):
    def to_rotation(data):
        t = data[:3]
        R = Rotation.from_quat(data[3:7]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    a = np.loadtxt(trajectory_path)
    out_dict = {i[0]: to_rotation(i[1:8]) for i in a}
    timestamps = np.array(sorted(out_dict.keys()))
    return out_dict, timestamps


import os.path as op
from easydict import EasyDict
import yaml
import numpy as np
from scipy.spatial.transform import Rotation


def read_Tel_from_yaml_file(yaml_file):
    cal_data = EasyDict()
    with open(yaml_file) as f:
        rig_data = yaml.load(f, Loader=yaml.FullLoader)
        rig_data = EasyDict(rig_data)
        cal_data = rig_data.rig
    lidar_ext = cal_data['lidar1']['extrinsic']
    tx, ty, tz, qx, qy, qz, qw = lidar_ext
    T = np.eye(4)
    q = Rotation.from_quat([qx, qy, qz, qw])
    if hasattr(q, "as_dcm"):
        T[:3, :3] = q.as_dcm().astype("float32")
    else:
        T[:3, :3] = q.as_matrix().astype("float32")
    T[:3, 3] = np.array([tx, ty, tz])
    return T


def transform_pts_with_T(points: np.ndarray, ego_to_lidar: np.ndarray) -> np.ndarray:
    points = np.concatenate([points, np.ones(points.shape[0])[:, None]], axis=1)
    camera_coords = points @ ego_to_lidar.T
    return camera_coords[:, :3]


def change_lidar_to_ego(pts):
    """
    Args:
        pts (np N*3): 输入3d点 在ego系
    output:
        pts: np N*3, ego系
    """
    scene_root = '/ssd1/data/4d/20230823_110018'  
    T_es = read_Tel_from_yaml_file(op.join(scene_root, "calibration.yml"))  
    pts_ego = transform_pts_with_T(pts, T_es)  
    return pts_ego

