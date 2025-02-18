from typing import Tuple, Dict
from pathlib import Path
from functools import partial
import yaml
from dataclasses import dataclass, fields

import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R

# from .helpers import find_n_closest_frames

EPS_FLOAT32 = float(np.finfo(np.float32).eps)
MAX_FLOAT32 = float(np.finfo(np.float32).max)


def to_homo(points: np.array) -> np.array:
    if points.ndim == 1:
        points = points[None, :]
    ones = np.ones((len(points), 1), dtype=np.float32)
    return np.concatenate((points, ones), axis=1)



class Calibration:
    def __init__(self, calib_info: dict) -> None:
        self._data = calib_info

    @classmethod
    def from_path(cls, path: Path) -> dict:
        with open(path, "r") as f:
            calib = yaml.safe_load(f)
        return cls(calib)


@dataclass
class Pose:
    timestamp: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    def __post_init__(self) -> None:
        for field in fields(self):
            setattr(self, field.name, field.type(getattr(self, field.name)))

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def quat(self) -> Tuple[float, float, float, float]:
        return (self.qx, self.qy, self.qz, self.qw)

    @property
    def Rt4x4(self) -> np.ndarray:
        return to_rt4x4(self.position, self.quat)


def to_rt4x4(position: Tuple[float, float, float], quat: Tuple[float, float, float, float]) -> np.ndarray:
    mat4x4 = np.eye(4, dtype=np.float32)
    mat4x4[:3, :3] = R.from_quat(quat).as_matrix()
    mat4x4[:3, 3] = position
    return mat4x4


class Trajectory:
    def __init__(self, poses: Dict[float, Pose]) -> None:
        self.poses = poses

    @classmethod
    def from_path(cls, path: Path) -> "Trajectory":
        poses = {}
        with path.open() as file:
            for line in file:
                timestamp, x, y, z, qx, qy, qz, qw = map(float, line.strip().split())
                pose = Pose(timestamp, x, y, z, qx, qy, qz, qw)
                poses[timestamp] = pose
        return cls(poses)

    def get_pose(self, timestamp: float) -> Pose:
        return self.poses[timestamp]



class FisheyeCameraModel:
    def __init__(self, calib, camera_id):
        camera_calib = calib["rig"][camera_id]
        self.pp = camera_calib["pp"]
        self.focal = camera_calib["focal"]
        self.inv_poly = np.array(camera_calib["inv_poly"], dtype=np.float32)
        self.image_size = camera_calib["image_size"]
        self.fov = camera_calib["fov_fit"] if 'fov_fit' in camera_calib.keys() else 200
        self.intrinsic_mat = np.array(
            [
                [self.focal[0], 0, self.pp[0]],
                [0, self.focal[1], self.pp[1]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def fit_unproj_func(p0, p1, p2, p3, fov=200):
        def proj_func(x, params):
            p0, p1, p2, p3 = params
            return x + p0 * x ** 3 + p1 * x ** 5 + p2 * x ** 7 + p3 * x ** 9

        def poly_odd6(x, k0, k1, k2, k3, k4, k5):
            return x + k0 * x ** 3 + k1 * x ** 5 + k2 * x ** 7 + k3 * x ** 9 + k4 * x ** 11 + k5 * x ** 13

        theta = np.linspace(-0.5 * fov * np.pi / 180, 0.5 * fov * np.pi / 180, 2000)
        theta_d = proj_func(theta, (p0, p1, p2, p3))
        params, pcov = curve_fit(poly_odd6, theta_d, theta)
        error = np.sqrt(np.diag(pcov)).mean()
        assert error < 1e-3, "poly parameter curve fitting failed: {:f}.".format(error)
        k0, k1, k2, k3, k4, k5 = params
        return partial(poly_odd6, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4, k5=k5)

    def unproject_points(self, points):
        unproj_func = self.fit_unproj_func(*self.inv_poly[:4])
        cx, cy = self.pp
        fx, fy = self.focal
        u = points[:, 0]
        v = points[:, 1]
        x_distorted = (u - cx) / fx
        y_distorted = (v - cy) / fy
        r_distorted = theta_distorted = np.sqrt(x_distorted ** 2 + y_distorted ** 2)
        r_distorted[r_distorted < 1e-5] = 1e-5
        theta = unproj_func(r_distorted)
        theta = np.clip(theta, - 0.5 * self.fov * np.pi / 180, 0.5 * self.fov * np.pi / 180)
        vignette_mask = np.float32(np.abs(theta * 180 / np.pi) < self.fov / 2)
        # camera coords on a sphere x-y-z right-down-forward
        dd = np.sin(theta)
        xx = x_distorted * dd / r_distorted
        yy = y_distorted * dd / r_distorted
        zz = np.cos(theta)
        fisheye_cam_coords = np.stack([xx, yy, zz], axis=1)
        return fisheye_cam_coords

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            points (np.ndarray): n*3 np

        Returns:
            np.ndarray: n*2 np
        """
        # 涉及到的参数，1 fov 2 inv_poly 3 intrinsic: focal /pp
        xc = points[:, 0]
        yc = points[:, 1]
        zc = points[:, 2]
        norm = np.sqrt(xc ** 2 + yc ** 2)
        theta = np.arctan2(norm, zc)
        fov_mask = theta > self.fov / 2 * np.pi / 180
        rho = (
                theta
                + self.inv_poly[0] * theta ** 3
                + self.inv_poly[1] * theta ** 5
                + self.inv_poly[2] * theta ** 7
                + self.inv_poly[3] * theta ** 9
        )
        width, height = self.image_size
        image_radius = np.sqrt((width / 2) ** 2 + (height) ** 2)
        rho[fov_mask] = 2 * image_radius / self.focal[0]
        xn = rho * xc / norm
        yn = rho * yc / norm
        xn[norm < EPS_FLOAT32] = 0
        yn[norm < EPS_FLOAT32] = 0
        norm_coords = np.stack([xn, yn, np.ones_like(xn)], axis=1)
        intrinsic_mat = np.array(
            [
                [self.focal[0], 0, self.pp[0]],
                [0, self.focal[1], self.pp[1]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        image_coords = norm_coords @ intrinsic_mat.T
        return image_coords[:, :2]

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.fisheye.undistortImage(
            distorted=image,
            K=self.intrinsic_mat,
            D=self.inv_poly[:4],
            Knew=self.intrinsic_mat
        )

    def undistort_points(self, points: np.ndarray, out_as_homo=True) -> np.ndarray:
        """Undistort 2D points on distorted image to 2D points on undistorted image."""
        norm_coords_homo = self.unproject_points(points)
        undistorted_points_on_image = norm_coords_homo @ self.intrinsic_mat.T  # 用小孔相机（假设无畸变）的方式将相机系中的3D点投到图像上
        if not out_as_homo:
            undistorted_points_on_image = undistorted_points_on_image[:, :2] / undistorted_points_on_image[:, 2:]
        return undistorted_points_on_image

    # def unproject_points(self, points: np.ndarray) -> np.ndarray:
    #     """Undistort 2D points on distorted image to 3D points in camera coordinate system (with Z guessed, Z=1)."""
    #     norm_coords = cv2.undistortPoints(
    #         points.reshape(1, -1, 2),
    #         self.intrinsic_mat,
    #         self.inv_poly,
    #     )
    #     return np.squeeze(cv2.convertPointsToHomogeneous(norm_coords))


def get_camera_models(calib_path, cameras=None):
    calib = Calibration.from_path(calib_path)
    cameras = [s for s in calib._data['rig'].keys() if 'cam' in s] if cameras is None else cameras
    camera_models = {cam_id: FisheyeCameraModel(calib._data, cam_id) for cam_id in cameras}
    return camera_models

def to_camera_xy(calib_path, cam_id, points):
    camera = FisheyeCameraModel(Calibration.from_path(calib_path)._data, cam_id)
    return camera.project_points(points)

