import copy
import torch
from tqdm import tqdm
import torch.multiprocessing as torch_mp
import json
from pathlib import Path as P
import numpy as np
from typing import List, Union
import yaml
from easydict import EasyDict
from scipy.spatial.transform import Rotation
import time
import os
import re
import os.path as op
import subprocess
from pathlib import Path
import multiprocessing as mp
EPS_FLOAT32 = float(np.finfo(np.float32).eps)
MAX_FLOAT32 = float(np.finfo(np.float32).max)

ipm_cls_dict = {
    0: "lanemark",
    1: "parkingline",
    2: "deceleraion_hump",
    3: "warningpole",
    4: "warningcone",
    5: "guideline",
    6: "groundpin",
    7: "groundlock",
    8: "zebracrosswalk",
}

label_to_class = {
    "101": "Box_truck",
    "102": "Truck",
    "103": "Car",
    "104": "Van",
    "105": "Bus",
    "106": "Engineering_vehicle",
    "201": "Pedestrian",
    "202": "Cyclist",
    "301": "Bicycle",
    "100": "DontCare",
}
velocitys_threshold_dict = {
    "Box_truck": 1,
    "Car": 1,
    "Pedestrian": 0.5,
    "Cyclist": 0.5,
    "Van": 1,
    "Truck": 1,
    "Bus": 1,
    "Engineering_vehicle": 1,
    "Bicycle": 0.5,
    "DontCare": 10,
}  # m/s

rotate_class = ["Box_truck", "Van", "Car", "Truck", "Bus", "Cyclist", "Bicycle", "Engineering_vehicle"]

min_l_dict = {
    "Box_truck": 3,
    "Car": 3,
    "Pedestrian": 0.1,
    "Cyclist": 1,
    "Van": 3,
    "Truck": 3,
    "Bus": 3,
    "Engineering_vehicle": 3,
    "Bicycle": 1,
    "DontCare": 0.1,
}  # m

min_points_dict = {
    "Box_truck": 10,
    "Car": 10,
    "Pedestrian": 5,
    "Cyclist": 5,
    "Van": 10,
    "Truck": 10,
    "Bus": 10,
    "Engineering_vehicle": 10,
    "Bicycle": 5,
    "DontCare": 5,
}

side_cameras = ["sv_left", "sv_right"]
main_cameras = ["sv_front", "sv_rear"]

sensor_time_offset = {"camera_sv_front": -50, "camera_sv_left": -50, "camera_sv_right": -50, "camera_sv_rear": -50}


def generate_lidar_mask(points, img_shape, CameraExtrinsicMat, CameraMat, DistCoeff, filter_xy=True, use_gpu=True, fov=200, round_pixel=True):
    """
    only support opencv camera;  points_raw: lidar系， img_shape, sensor2lidar，camera_K, D
    """
    is_numpy = False
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()
        is_numpy = True
    if isinstance(img_shape, np.ndarray):
        img_shape = torch.from_numpy(img_shape).float()
    elif isinstance(img_shape, list) or isinstance(img_shape, tuple):
        img_shape = torch.from_numpy(np.array(img_shape)).float()
    if isinstance(CameraExtrinsicMat, np.ndarray):
        CameraExtrinsicMat = torch.from_numpy(CameraExtrinsicMat).float()
    if isinstance(CameraMat, np.ndarray):
        CameraMat = torch.from_numpy(CameraMat).float()
    if isinstance(DistCoeff, np.ndarray):
        DistCoeff = torch.from_numpy(DistCoeff).float()
    elif isinstance(DistCoeff, list):
        DistCoeff = torch.from_numpy(np.array(DistCoeff)).float()
    device = points.device
    if use_gpu:
        points = points.cuda()
        img_shape = img_shape.cuda()
        CameraExtrinsicMat = CameraExtrinsicMat.cuda()
        CameraMat = CameraMat.cuda()
        DistCoeff = DistCoeff.cuda()
    assert len(DistCoeff) in [4, 5]
    p = points[:, :3]
    # p=np.hstack((p,np.zeros((p.shape[0],1))+1))
    p = torch.cat([p, torch.ones((p.shape[0], 1), device=p.device)], dim=1)
    # camera_coordinates=np.dot(CameraExtrinsicMat,p.T)
    camera_coordinates = torch.matmul(CameraExtrinsicMat, p.T)
    # index = np.argsort(-camera_coordinates[2,:])
    # camera_coordinates = camera_coordinates[:,index]
    # front_filter = camera_coordinates[2,:]>0
    # camera_coordinates=camera_coordinates[:,camera_coordinates[2,:]>0]
    camera_coords = camera_coordinates.T
    if len(DistCoeff) == 5:
        fov_mask = camera_coordinates[2, :] < 0
        normalize_coordinates = camera_coordinates / (camera_coordinates[2, :] + EPS_FLOAT32)
        xc = copy.deepcopy(normalize_coordinates[0, :])
        yc = copy.deepcopy(normalize_coordinates[1, :])
        r2 = xc * xc + yc * yc
        tmpdist = 1 + DistCoeff[0] * r2 + DistCoeff[1] * r2 * r2 + DistCoeff[4] * r2 * r2 * r2
        normalize_coordinates[0, :] = xc * tmpdist + 2 * DistCoeff[2] * xc * yc + DistCoeff[3] * (r2 + 2 * xc * xc)
        normalize_coordinates[1, :] = yc * tmpdist + DistCoeff[2] * (r2 + 2 * yc * yc) + 2 * DistCoeff[3] * xc * yc
    elif len(DistCoeff) == 4:
        xc = copy.deepcopy(camera_coordinates[0, :])
        yc = copy.deepcopy(camera_coordinates[1, :])
        zc = copy.deepcopy(camera_coordinates[2, :])
        r2 = xc * xc + yc * yc
        norm = torch.sqrt(r2)
        theta = torch.atan2(norm, zc)
        FOV = fov
        FOV_ = FOV / 2 * np.pi / 180
        fov_mask = theta > FOV_
        rho = theta + DistCoeff[0] * theta**3 + DistCoeff[1] * theta**5 + DistCoeff[2] * theta**7 + DistCoeff[3] * theta**9
        image_radius = torch.sqrt((img_shape[1] / 2) ** 2 + (img_shape[0]) ** 2)
        focal = [CameraMat[0, 0], CameraMat[1, 1]]
        # if filter_xy:
        rho[fov_mask] = 2 * image_radius / focal[0]
        # else:
        #     rho[fov_mask] = FOV_ + DistCoeff[0] * FOV_**3 + DistCoeff[1] * FOV_**5 + \
        #                 DistCoeff[2] * FOV_**7 + DistCoeff[3] * FOV_**9
        xn = rho * xc / norm
        yn = rho * yc / norm
        xn[norm < EPS_FLOAT32] = 0
        yn[norm < EPS_FLOAT32] = 0
        # normalize_coordinates = np.concatenate([np.vstack([xn, yn]), np.ones((1,xn.shape[0]))],axis=0)
        normalize_coordinates = torch.cat([torch.vstack([xn, yn]), torch.ones((1, xn.shape[0]), device=xn.device)], dim=0)

    # pixel_coordinates=np.dot(CameraMat[:3,:3],normalize_coordinates)
    pixel_coordinates = torch.matmul(CameraMat[:3, :3], normalize_coordinates)
    # z_filter=pixel_coordinates[2,:]>0
    x_filter = torch.logical_and(pixel_coordinates[0, :] > 0, pixel_coordinates[0, :] < img_shape[1])
    y_filter = torch.logical_and(pixel_coordinates[1, :] > 0, pixel_coordinates[1, :] < img_shape[0])
    all_filter = x_filter & y_filter & ~fov_mask
    if filter_xy:
        pixel_coordinates = pixel_coordinates[:, all_filter]
        camera_coords = camera_coords[all_filter]
        # if len(DistCoeff)==5:
        #     all_filter = x_filter & y_filter & ~fov_mask
        # elif len(DistCoeff)==4:
        #     all_filter = x_filter & y_filter & ~fov_mask
    # else:
    # if len(DistCoeff)==5:
    #     all_filter = torch.ones_like(z_filter).bool()
    # elif len(DistCoeff)==4:
    #     all_filter = torch.ones_like(z_filter).bool()
    # all_filter = torch.ones_like(fov_mask).bool()
    pixel_coordinates = pixel_coordinates.T
    if round_pixel:
        pixel_coordinates = pixel_coordinates.long()
    img_shape = img_shape.long().cpu().numpy()
    mask = torch.zeros((img_shape[0], img_shape[1], 3), device=camera_coords.device)
    if filter_xy:
        mask[pixel_coordinates[:, 1], pixel_coordinates[:, 0]] = camera_coords[:, :3]
        if is_numpy:
            return (mask.cpu().numpy(), all_filter.cpu().numpy(), pixel_coordinates.cpu().numpy())
        else:
            return (mask.to(device), all_filter.to(device), pixel_coordinates.to(device))
    else:
        if is_numpy:
            return (all_filter.cpu().numpy(), pixel_coordinates.cpu().numpy())
        else:
            return (all_filter.to(device), pixel_coordinates.to(device))


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


def anno_box_to_7_values_box(box: dict) -> np.ndarray:
    return np.array(
        [
            [
                box["psr"]["position"]["x"],
                box["psr"]["position"]["y"],
                box["psr"]["position"]["z"],
                box["psr"]["scale"]["x"],
                box["psr"]["scale"]["y"],
                box["psr"]["scale"]["z"],
                box["psr"]["rotation"]["z"],
            ]
        ],
        dtype=np.float32,
    )

def anno_box_to_9_values_box(box: dict) -> np.ndarray:
    return np.array(
        [
            [
                box["psr"]["position"]["x"],
                box["psr"]["position"]["y"],
                box["psr"]["position"]["z"],
                box["psr"]["scale"]["x"],
                box["psr"]["scale"]["y"],
                box["psr"]["scale"]["z"],
                box["psr"]["rotation"]["x"],
                box["psr"]["rotation"]["y"],
                box["psr"]["rotation"]["z"],
            ]
        ],
        dtype=np.float32,
    )

def read_json_to_list(path):
    with open(str(path)) as f:
        a = json.load(f)
    return a
def read_json(path):
    with open(str(path)) as f:
        a = json.load(f)
    return a


def format_floats(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = format_floats(v)
        return obj
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = format_floats(v)
        return obj
    else:
        return obj


def write_json_from_list(output_dict, save_path, format_float=False, indent=None):
    save_path = str(save_path)
    P(save_path).parent.mkdir(exist_ok=True, parents=True)
    if format_float:
        output_dict = format_floats(output_dict)
    with open(save_path, "w") as f:
        # json.dump(output_dict, f, indent=4, default=lambda x: round(x, 3))
        json.dump(output_dict, f, default=lambda x: round(x, 3), indent=indent)


def transform_points_ego_to_lidar(points: np.ndarray, ego_to_lidar: np.ndarray) -> np.ndarray:
    assert points.ndim == 2
    if points.shape[1] == 3:
        points = np.concatenate([points, np.ones(points.shape[0])[:, None]], axis=1)
    camera_coords = points @ ego_to_lidar.T
    return camera_coords[:, :3]

class CameraParam:
    def __init__(self, calib: dict, middle_lidar_id: str, camera_id: str) -> None:
        self.lidar_to_ego = calib[middle_lidar_id]["T_es"]
        self.ego_to_lidar = np.linalg.inv(self.lidar_to_ego)
        self.cam_to_ego = calib[camera_id]["T_es"]
        self.ego_to_cam = np.linalg.inv(self.cam_to_ego)
        self.lidar_to_cam = np.dot(np.linalg.inv(self.cam_to_ego), self.lidar_to_ego)
        self.cam_to_lidar = np.linalg.inv(self.lidar_to_cam)
        self.image_shape = calib[camera_id]["image_size"][::-1]
        self.K = calib[camera_id]["K"]
        self.D = calib[camera_id]["D"]
        self.camera_id = camera_id
        self.fov = calib[camera_id]['fov_fit']
        self.inv_poly = calib[camera_id]['inv_poly']
        self.image_width, self.image_height = calib[camera_id]['image_size']
        self.image_radius = np.sqrt((self.image_width / 2) ** 2 + (self.image_height) ** 2)

def check_whether_points_behind_camera(points: torch.Tensor, ego_to_cam: torch.Tensor) -> torch.Tensor:
    if points.ndim == 2 and points.shape[1] == 3:
        points = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=1)
    camera_coords = points @ ego_to_cam.T
    return camera_coords[:, 2] < 0


def find_path_from_ts_and_dir_path(ts, dir_path):
    times, file_names = get_times(dir_path)
    return op.join(dir_path, get_sync_filename(file_names, times, ts))


def find_path_from_ts_and_dir_path_maybe_not_find(ts, dir_path):
    times, file_names = get_times(dir_path)
    if get_sync_filename(file_names, times, ts) is None:
        return None
    return op.join(dir_path, get_sync_filename(file_names, times, ts))



def read_ts_json(path):
    ts_dict_list = read_json_to_list(path)
    output_dict = {i["lidar"]: i for i in ts_dict_list}
    ts_list = sorted([i["lidar"] for i in ts_dict_list])
    return ts_list, output_dict


def torch_pool(func, list_to_process, n_processes=torch_mp.cpu_count()):
    torch_mp.set_start_method("spawn", force=True)
    with  torch_mp.Pool(n_processes) as pool:
        r = list(tqdm(pool.imap(func, list_to_process), total=len(list_to_process)))
    return r

def mp_pool(func, list_to_process, n_processes=mp.cpu_count()):
    with mp.Pool(n_processes) as p:
        r = list(tqdm(p.imap(func, list_to_process), total=len(list_to_process)))
    return r
