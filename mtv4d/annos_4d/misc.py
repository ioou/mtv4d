import copy

import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R, Rotation
from scipy.spatial import Delaunay
from pathlib import Path as P
import os.path as op



def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)
    return flag

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.vstack([
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ]).T.reshape(-1, 3, 3).astype(np.float32)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate([points_rot, points[:, :, 3:]], axis=-1)
    return points_rot


def boxes_to_corners_3d(boxes3d):
    """
    lidar cor:
           4 -------- 5
        /|         /|
        7 -------- 6 .
        | |        | |
        . 0 -------- 1
        |/         |/
        3 -------- 2
    Args:
        boxes3d:  (N, 7) [x, y, z, l, w, h, heading], (x, y, z) is the box center
    Returns:
        corners3d: (N, 8, 3)
    """
    template = np.array([
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1]
    ]) / 2
    corners3d = np.repeat(boxes3d[:, None, 3:6], 8, axis=1) * template
    corners3d = rotate_points_along_z(
        corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def translate_yprt_from_world_to_frame_deprecated(box, Twe):
    # written by me and find the euler not that way
    x = box["psr"]["position"]["x"]
    y = box["psr"]["position"]["y"]
    z = box["psr"]["position"]["z"]
    x_ = box["psr"]["rotation"]["x"]
    y_ = box["psr"]["rotation"]["y"]
    z_ = box["psr"]["rotation"]["z"]
    R = Rotation.from_euler('zyx', [z_, y_, x_]).as_matrix()
    t = [x, y, z]
    Tow = np.eye(4)
    Tow[:3, :3] = R
    Tow[:3, 3] = t
    Toe = Tow @ Twe
    rotation_ego = Rotation.from_matrix(Tow[:3, :3]).as_euler('zyx')
    return {
        'position': Toe[:3, 3].tolist(),
        'scale': [box["psr"]["scale"]['x'], box["psr"]["scale"]['y'], box["psr"]["scale"]['z']],
        'rotation': rotation_ego.tolist()
    }


def translate_and_duplicate_pts3d_with_T(polyline, T):
    new_polyline = copy.deepcopy(polyline)
    new_polyline['vertices'] = translate_pts3d_with_T(new_polyline['vertices'], T)
    return new_polyline

def translate_pts3d_with_T(pts3d, T):
    pts3d = np.array(pts3d).reshape(-1, 3)
    if len(pts3d)==0:
        return pts3d
    pts4d = np.concatenate([pts3d, np.ones([pts3d.shape[0], 1])], axis=1)
    new_pts = (T@pts4d.T).T
    return new_pts[:, :3]

def translate_box9d_with_T(bx, Tts):
    # Tts 表示 从box的当前系转到目标系
    position = np.array([[bx["psr"]["position"]["x"], bx["psr"]
                        ["position"]["y"], bx["psr"]["position"]["z"], 1.0]])
    rotation = np.eye(4)
    rotation[:3, :3] = R.from_euler("XYZ", [bx["psr"]["rotation"]["x"], bx["psr"]["rotation"]["y"],
                                            bx["psr"]["rotation"]["z"]]).as_matrix()
    transformed_position = (position @ Tts.T).squeeze()
    transformed_rotation = R.from_matrix(
        (Tts @ rotation)[:3, :3]).as_euler("XYZ")
    transformed_box = copy.deepcopy(bx)
    transformed_box["psr"]["position"] = {
        "x": transformed_position[0],
        "y": transformed_position[1],
        "z": transformed_position[2],
    }
    transformed_box["psr"]["rotation"] = {
        "x": transformed_rotation[0],
        "y": transformed_rotation[1],
        "z": transformed_rotation[2],
    }
    return transformed_box

def translate_psr_with_T(psr, Tts):
    # Tts 表示 从box的当前系转到目标系
    position = np.array([[psr["position"]["x"],psr["position"]["y"], psr["position"]["z"], 1.0]])
    rotation = np.eye(4)
    rotation[:3, :3] = R.from_euler("XYZ", [psr["rotation"]["x"], psr["rotation"]["y"],
                                            psr["rotation"]["z"]]).as_matrix()
    transformed_position = (position @ Tts.T).squeeze()
    transformed_rotation = R.from_matrix(
        (Tts @ rotation)[:3, :3]).as_euler("XYZ")
    new_psr = copy.deepcopy(psr)
    new_psr["position"] = {
        "x": transformed_position[0],
        "y": transformed_position[1],
        "z": transformed_position[2],
    }
    new_psr["rotation"] = {
        "x": transformed_rotation[0],
        "y": transformed_rotation[1],
        "z": transformed_rotation[2],
    }
    return new_psr



def translate_psr_to_output_geometry(psr):
    output = {}
    output['pos_xyz'] = [psr['position']['x'], psr['position']['y'], psr['position']['z']]
    output['scale_xyz'] = [psr['scale']['x'], psr['scale']['y'], psr['scale']['z']]
    output['rot_xyz'] = [psr['rotation']['x'], psr['rotation']['y'], psr['rotation']['z']]
    return output
def translate_output_geometry_to_psr(box):
    output = {
        'position': {'x': box['pos_xyz'][0],
                     'y':box['pos_xyz'][1],
                     'z':box['pos_xyz'][2],
                     },
        'scale': {'x': box['scale_xyz'][0],
                     'y':box['scale_xyz'][1],
                     'z':box['scale_xyz'][2],
                     },
        'rotation': {'x': box['rot_xyz'][0],
                     'y':box['rot_xyz'][1],
                     'z':box['rot_xyz'][2],
                     },
    }
    return output

def read_pose_txt(pose_txt_path):
    with open(pose_txt_path, 'r') as f:
        data = f.readlines()
    time = []
    x = []
    y = []
    z = []
    q = []
    T = []
    frame = []
    for i, line in enumerate(data):
        d_line = line.split(' ')
        if d_line[0] == 'time':
            continue
        if len(time) > 0:
            if float(d_line[0]) == time[-1]:
                continue
        time.append(float(d_line[0]))
        x = float(d_line[1])
        y = float(d_line[2])
        z = float(d_line[3])
        q = np.array([float(d_line[4]), float(d_line[5]),
                     float(d_line[6]), float(d_line[7])])
        # frame.append(float(d_line[8]))
        frame.append(i)
        rot = R.from_quat(np.array(q)).as_matrix()
        T_ = np.eye(4)
        T_[:3, :3] = rot
        T_[:3, 3] = np.array([x, y, z])
        T.append(T_)
    Twes = np.stack(T, axis=0)
    return np.array(time), Twes, frame


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



