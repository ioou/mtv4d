
# 最后一次重复造轮子
from scipy.spatial.transform import Rotation
import os.path as op
import numpy as np
from pyquaternion import Quaternion
from pathlib import Path as P


def homo43(pts):
    return np.concatenate([pts, np.ones([len(pts), 1])], axis=1)


def to_corners_7(pts):
    """

    lidar cor:
       2 -------- 3
      /|         /|
    6 -------- 7 .
    | |        | |
    . 1 -------- 0
    |/         |/
    5 -------- 4

    """
    x, y, z = pts[:3]
    a, b, c = pts[3:6] / 2
    # u, v, w = pts[6:9]
    u, v, w = 0, 0, pts[6]
    corners = np.array(
        [
            [+a, -b, -c],
            [+a, +b, -c],
            [+a, +b, +c],
            [+a, -b, +c],
            [-a, -b, -c],
            [-a, +b, -c],
            [-a, +b, +c],
            [-a, -b, +c],
        ]
    )
    Rx = Rotation.from_euler("x", u).as_matrix()
    Ry = Rotation.from_euler("y", v).as_matrix()
    Rz = Rotation.from_euler("z", w).as_matrix()
    R = Rx @ Ry @ Rz
    corners = (R @ corners.T).T
    corners += np.array([x, y, z]).reshape(-1, 3)
    return corners


def to_corners_9(pts):
    """

    lidar cor:
       2 -------- 3
      /|         /|
    6 -------- 7 .
    | |        | |
    . 1 -------- 0
    |/         |/
    5 -------- 4

    """
    x, y, z = pts[:3]
    a, b, c = pts[3:6] / 2
    # u, v, w = pts[6:9]
    u, v, w = 0, 0, pts[6]
    corners = np.array(
        [
            [+a, -b, -c],
            [+a, +b, -c],
            [+a, +b, +c],
            [+a, -b, +c],
            [-a, -b, -c],
            [-a, +b, -c],
            [-a, +b, +c],
            [-a, -b, +c],
        ]
    )
    Rx = Rotation.from_euler("x", u).as_matrix()
    Ry = Rotation.from_euler("y", v).as_matrix()
    Rz = Rotation.from_euler("z", w).as_matrix()
    R = Rx @ Ry @ Rz
    corners = (R @ corners.T).T
    corners += np.array([x, y, z]).reshape(-1, 3)
    return corners


def draw_boxes(img, corners2d, text_list=None):
    if text_list is None:
        text_list = np.arange(len(corners2d.reshape(-1, 8, 2)))
    for idx, (pts, txt) in enumerate(zip(corners2d.reshape(-1, 8, 2),text_list)):
        pt = pts[:, :2].astype('int')
        # if (pt < 0).any() or (pt > 1600).any():  continue
        cv2.polylines(img, [pt[:4]], 2, (0, 255, 255), 2)
        cv2.polylines(img, [pt[4:]], 2, (0, 0, 255), 2)
        for i in range(4):
            cv2.line(img, tuple(pt[i]), tuple(pt[i + 4]), (0, 0, 255), 2)
        # if int(txt) < 20:
        if txt:
            cv2.putText(img, str(txt), tuple(pt[0]), 1, 2, (255, 255,0),2)
    return img


import cv2
import matplotlib.pyplot as plt

EPS_FLOAT32 = float(np.finfo(np.float32).eps)
MAX_FLOAT32 = float(np.finfo(np.float32).max)


def project_points_fisheye(points, T_c, inv_poly, image_size, intrinsic_mat, fov, focal):
    # 涉及到的参数，1 fov 2 inv_poly 3 intrinsic: focal /pp
    pts4d = homo43(points) @ T_c.T
    points = pts4d
    xc = points[:, 0]
    yc = points[:, 1]
    zc = points[:, 2]
    norm = np.sqrt(xc ** 2 + yc ** 2)
    theta = np.arctan2(norm, zc)
    fov_mask = theta > fov / 2 * np.pi / 180
    rho = (
            theta
            + inv_poly[0] * theta ** 3
            + inv_poly[1] * theta ** 5
            + inv_poly[2] * theta ** 7
            + inv_poly[3] * theta ** 9
    )
    width, height = image_size
    image_radius = np.sqrt((width / 2) ** 2 + (height) ** 2)
    rho[fov_mask] = 2 * image_radius / focal[0]
    xn = rho * xc / norm
    yn = rho * yc / norm
    xn[norm < EPS_FLOAT32] = 0
    yn[norm < EPS_FLOAT32] = 0
    norm_coords = np.stack([xn, yn, np.ones_like(xn)], axis=1)

    image_coords = norm_coords @ intrinsic_mat[:, :3].T
    return image_coords[:, :2]


def project_points_ordinary_cam(pts3d, T_c, intrinsic_mat):
    pts4d = homo43(pts3d) @ T_c.T
    corners3d = pts4d[:, :3] / pts4d[:, 2:3]
    pts_cam = (intrinsic_mat @ corners3d.T).T
    return pts_cam[:, :2]


def draw_points_to_cams(im, boxes, Tce, intrinsic_mat, inv_poly, save_im=True):
    """
    T_c   : T_cl  #
    boxes: 7个数
    """
    image_size = im.shape[:2]
    pts3d = np.concatenate([to_corners_7(b) for b in boxes])  # n8 * 3
    if inv_poly is None:
        pts2d = project_points_ordinary_cam(pts3d, Tce, intrinsic_mat)
    else:
        pts2d = project_points_fisheye(pts3d, Tce, inv_poly, im.shape[:2], intrinsic_mat, 200,
                                       [intrinsic_mat[0, 0], intrinsic_mat[1, 1]])
    draw_boxes(im, pts2d)
    from time import time
    path = f'/tmp/1234/3/{time()}.jpg'
    if save_im:
        result = cv2.imwrite(path, im)
    return im




