import open3d as o3d
import numpy as np
from pathlib import Path as P
import os.path as op
from mtv4d.utils.io_base import read_points_from_bin, write_bin

from scipy.spatial import Delaunay
from pathlib import Path as P
import scipy


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


def hpr_index(points, radius=None, view_point=None):
    radius = radius or np.abs(points[:, :3]).max() * 100
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    viewpoint = np.array([0, 0, 0]) if view_point is None else np.array(view_point)
    _, pt_map = pcd.hidden_point_removal(viewpoint, radius)
    return pt_map


def open3d_hpr(points, radius=3000.0, view_point=None):
    pt_map = hpr_index(points[:, :3], radius=radius, view_point=view_point)
    return points[np.array(sorted(pt_map))]


def hidden_point_generate(path):
    root_path = P(path.parent.parent.parent)
    points = read_points_from_bin(str(path))
    points = open3d_hpr(points)
    save_path = op.join(root_path, "hidden_point_removal/lidar1", path.name)
    write_bin(points, save_path)


def pick_visible_points_according_to_hpr_points(poly_points, hpr_points, thres=0.2):
    import torch
    distances = torch.cdist(torch.tensor(poly_points).double(), torch.tensor(hpr_points).double())
    idx = torch.argsort(distances, axis=1)
    dist_smallest = torch.take_along_dim(distances, idx[:, :1], axis=1)[:, 0]
    # dist_smallest = distances[:, idx[:, 0]]
    a = dist_smallest < thres
    return "".join(["1" if i else "0" for i in a])


def draw_points(points, save_path='output.ply'):
    # need points
    points = points[:, :3]
    add_points = np.zeros([300, 3])
    add_points[:100, 0] = np.arange(100) / 10
    add_points[100:200, 1] = np.arange(100) / 10
    add_points[200:, 2] = np.arange(100) / 10
    points = np.concatenate([points, add_points])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    P(save_path).parent.mkdir(exist_ok=True, parents=True)
    o3d.io.write_point_cloud(save_path, pcd)


def draw_points_with_color(points, colors, save_path='output.ply'):
    # need points
    points = points[:, :3]
    add_points = np.zeros([300, 3])
    add_points[:100, 0] = np.arange(100) / 10
    add_points[100:200, 1] = np.arange(100) / 10
    add_points[200:, 2] = np.arange(100) / 10
    points = np.concatenate([points, add_points])
    colors = np.concatenate([colors, np.zeros_like(add_points)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    P(save_path).parent.mkdir(exist_ok=True, parents=True)
    o3d.io.write_point_cloud(save_path, pcd)


def load_pcd(pcd_path):
    lidar_data = o3d.t.io.read_point_cloud(pcd_path)
    lidar_points_positions = lidar_data.point.positions.numpy()
    lidar_points_intensity = lidar_data.point.intensity.numpy()
    lidar_points = np.concatenate([lidar_points_positions, lidar_points_intensity, np.zeros_like(lidar_points_intensity)], axis=1)
    return lidar_points


def write_pcd(points, save_path, intensity=False, color=False):
    pcd = o3d.t.geometry.PointCloud()
    device = o3d.core.Device("CPU:0")
    a = points.astype('float')
    dtype = o3d.core.float32
    pcd.point.positions = o3d.core.Tensor(a[:, :3], dtype)
    if intensity:
        pcd.point.intensity = o3d.core.Tensor(a[:, 3:4], dtype)
    if color:
        pcd.point.color = o3d.core.Tensor(a[:, 4:6], dtype)
    o3d.t.io.write_point_cloud(str(save_path), pcd)


def draw_occ(voxels):
    import torch
    def from_voxel_to_edges(voxels):
        bases_ = torch.arange(0, voxels.shape[0] * 8, 8)
        edges = torch.tensor([[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
        edges = edges.reshape((1, 12, 2)).repeat(voxels.shape[0], 1, 1)
        edges = np.array(edges + bases_[:, None, None])

        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(voxels.reshape((-1, 3)))
        line_sets.lines = o3d.open3d.utility.Vector2iVector(edges.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        return voxels, line_sets
    voxel_grid, line_sets = from_voxel_to_edges(voxels)
    vis = o3d.visualization.Visualizer()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    vis.add_geometry(voxel_grid)
    vis.add_geometry(line_sets)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 2])
    vis.add_geometry(mesh_frame)
    o3d.visualization.draw_geometries([line_sets])


if __name__ == "__main__":
    draw_occ(np.array([[0,0,0], [3,3,3]]))

