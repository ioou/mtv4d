import numpy as np


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
