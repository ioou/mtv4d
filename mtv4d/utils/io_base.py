import json
import pickle
import numpy as np
from pathlib import Path as P


def read_txt(path, to_format=True):
    with open(str(path)) as f:
        data = f.readlines()
    if to_format:
        data = [i.strip('\n') for i in data]
    return data


def write_txt(data, path, to_format=True):
    P(path).parent.mkdir(exist_ok=True, parents=True)
    if to_format:
        data = [i + '\n' for i in data]
    with open(str(path), 'w') as f:
        f.writelines(data)


def read_pickle(path):
    with open(str(path), 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(data, path):
    P(path).parent.mkdir(exist_ok=True, parents=True)
    with open(str(path), 'wb') as f:
        pickle.dump(data, f)


def read_json(path):
    with open(str(path)) as f:
        data = json.load(f)
    return data


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


def write_json(output_dict, save_path, format_float=False, indent=None):
    save_path = str(save_path)
    P(save_path).parent.mkdir(exist_ok=True, parents=True)
    if format_float:
        output_dict = format_floats(output_dict)
    with open(save_path, "w") as f:
        json.dump(output_dict, f, default=lambda x: round(x, 4), indent=indent)


def write_json_from_list(output_dict, save_path, format_float=False, indent=None, round_float=4):
    save_path = str(save_path)
    P(save_path).parent.mkdir(exist_ok=True, parents=True)
    if format_float:
        output_dict = format_floats(output_dict)
    with open(save_path, "w") as f:
        json.dump(output_dict, f, default=lambda x: round(x, round_float), indent=indent)


def read_points_from_pcd(lidar_path):
    import open3d as o3d
    lidar_data = o3d.t.io.read_point_cloud(lidar_path)
    lidar_points_positions = lidar_data.point.positions.numpy()
    lidar_points_intensity = lidar_data.point.intensity.numpy()
    lidar_points = np.concatenate([lidar_points_positions, lidar_points_intensity], axis=1)
    return lidar_points


def read_pcd(path, intensity) -> np.ndarray:
    """read pcd file

    Parameters
    ----------
    path : Union[str, Path]
        _description_
    intensity : bool, optional
        _description_, by default True
        
    Returns
    -------
    np.ndarray
        - if intensity is false, return (N, 3) array, i.e. [[x,y,z], ...]
        - if intensity is true, return (N, 4) array, i.e. [[x,y,z,intensity], ...]
    """
    import pypcd
    pcd = pypcd.PointCloud.from_path(path)
    npdata = np.stack([pcd.pc_data['x'], pcd.pc_data['y'], pcd.pc_data['z']], axis=1)
    if intensity:
        npdata = np.concatenate([npdata, pcd.pc_data['intensity'].reshape(-1, 1)], axis=1)
    return npdata


def read_points_from_bin(path):
    return np.fromfile(str(path), dtype='float32').reshape(-1, 4)


def write_bin(data, path):
    P(path).parent.mkdir(exist_ok=True, parents=True)
    data.tofile(str(path))


def write_ply(pts, path, clrs=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if clrs is not None:
        pcd.colors = o3d.utility.Vector3dVector(clrs)
    o3d.io.write_point_cloud(path, pcd)


def draw_points_with_coord_origin(points):
    import open3d as o3d
    add_points = np.zeros([300, 3])
    add_points[:100, 0] = np.arange(100) / 10
    add_points[100:200, 1] = np.arange(100) / 10
    add_points[200:, 2] = np.arange(100) / 10
    points = np.concatenate([points, add_points])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("output.ply", pcd)
