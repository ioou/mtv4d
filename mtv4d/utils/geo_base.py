import numpy as np
from scipy.spatial.transform import Rotation


def to_homo(points: np.array) -> np.array:
    if points.ndim == 1:
        points = points[None, :]
    ones = np.ones((len(points), 1), dtype=np.float32)
    return np.concatenate((points, ones), axis=1)


def transform_pts_with_T(points, T):
    # from pts3d to lidar 3d
    points = np.array(points)
    shape = points.shape
    points = points.reshape(-1, 3)
    points_output = (to_homo(points) @ T.T)[:, :3].reshape(*shape)
    return points_output

def Rt2T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def T2Rt(T):
    return T[:3, :3], T[:3, 3]

def rotation_to_wxyz(rot):
    R = np.array(rot).reshape(3, 3)
    return Rotation.from_matrix(R).as_quat(scalar_first=True).tolist()  # wxyz, default False

def rotation_to_xyzw(rot):
    R = np.array(rot).reshape(3, 3)
    return Rotation.from_matrix(R).as_quat(scalar_first=False).tolist()
