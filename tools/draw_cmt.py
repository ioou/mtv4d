from pathlib import Path as P
import numpy as np
from scipy.spatial.transform import Rotation

from mtv4d.utils.geo_base import transform_pts_with_T, Rt2T
from mtv4d.utils.draw_base import draw_boxes
from mtv4d.annos_4d.misc import read_ego_paths
from mtv4d.utils.box_base import to_corners_9
from mtv4d.utils.io_base import read_json, read_pickle
from mtv4d.utils.sensors import get_camera_models
from mtv4d.utils.calib_base import read_cal_data
import cv2

if __name__ == "__main__":
    import os.path as op

    data_root = "/ssd4/tmp/1"
    scene_id = "20231104_170321"
    sensor = "camera8"
    scene_root = f"{data_root}/{scene_id}"
    im_dir = f"{scene_root}/camera/camera8"
    cmt_pred_dir = f"{scene_root}/cmt"
    calib_path = f"{scene_root}/calibration_center.yml"
    timestamps = sorted([int(i.stem) for i in P(cmt_pred_dir).glob('*.json')])
    calib = read_cal_data(calib_path)
    camera_model = get_camera_models(calib_path, [sensor])[sensor]
    Twes, _ = read_ego_paths(op.join(scene_root, f"trajectory.txt"))
    scene_infos = read_pickle(f"{data_root}/mv_4d_infos_{scene_id}.pkl")[scene_id]
    Tes_f = Rt2T(scene_infos['scene_info']['calibration'][sensor]['extrinsic'][0],
                    scene_infos['scene_info']['calibration'][sensor]['extrinsic'][1])
    Tse_f = np.linalg.inv(Tes_f)
    Tse_r = calib[sensor]['T_se']

    T_fr = np.eye(4)
    T_fr[:3, :3] = Rotation.from_euler('z', [-np.pi / 2]).as_matrix()
    T_rf = np.linalg.inv(T_fr)


    def to_1(rotation):
        xyz = Rotation.from_matrix(np.array(rotation).reshape(3, 3)).as_euler("XYZ")
        return xyz.tolist()


    def f2r_list(box):
        Tf_box = np.eye(4)
        Tf_box[:3, :3] = Rotation.from_euler('XYZ', list(box['rotation'].values())).as_matrix()
        Tf_box[:3, 3] = np.array(list(box['position'].values()))
        Trf = np.eye(4)
        # Trf[:3, :3] = Rotation.from_euler('z', [np.pi / 2]).as_matrix()
        Tr_box = Trf @ Tf_box
        rotation = Rotation.from_matrix(Tr_box[:3, :3]).as_euler('XYZ')
        translation = Tr_box[:3, 3]
        return list(translation) + list(box['scale'].values()) + list(rotation)


    for ts in timestamps[200:250]:
        cmt_path = f"{cmt_pred_dir}/{ts}.json"
        a = read_json(cmt_path)
        im_path = f"{im_dir}/{ts}.jpg"
        im = cv2.imread(str(im_path))
        Tew = np.linalg.inv(Twes[float(ts)])

        if False:  # CMT, right-front-up

            boxes = read_json(cmt_path)


            def box_to_list(box):
                output = list(box['psr']['position'].values()) + list(box['psr']['scale'].values()) + list(
                    box['psr']['rotation'].values())
                return output
                # return box_psr['position'] + box_psr['scale'] + box_psr['rotation']


            box9d = [box_to_list(box_psr) for box_psr in boxes if
                     'class.vehicle.passenger_car' == box_psr['obj_type'] and max(
                         list(box_psr['psr']['position'].values())) < 15]

            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera
            Tew = np.linalg.inv(Twes[float(ts)])
            Tsw = Tse_r @ Tew
            corners3d = transform_pts_with_T(corners, Tse).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
        elif False:  # front-left-up    
            fb = "/home/yuanshiwei/4/prefusion/infer_results/pred_dumps/dets/20231104_170321"
            fb_path = f"{fb}/{ts}.json"
            data = read_json(fb_path)['pred']['bboxes']
            box9d = [i['translation'] + i['size'] + to_1(i['rotation']) for i in data]
            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera
            corners3d = transform_pts_with_T(corners, Tse_r @ T_rf).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
        elif False:
            fb = "/home/yuanshiwei/4/prefusion/1"
            fb_path = f"{fb}/{ts}.json"
            data = read_json(fb_path)
            # box9d = [i['translation'] + i['size'] + to_1(i['rotation']) for i in data]
            box9d = [f2r_list(i['psr']) for i in data]
            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera
            corners3d = transform_pts_with_T(corners, Tse_r).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
        elif False:
            psr = {
                'position': [-22.7799, -0.888688, 0.680343],
                "rotation": [-0.005630966435722584, -0.007195778542323339, 2.2345830262983197],
                "scale": [4.75924, 2.02294, 1.55872]
            }
            box9d = [psr['position'] + psr['scale'] + psr['rotation']]
            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera

            corners3d = transform_pts_with_T(corners, Tse_r @ Tew).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
        else:
            #     ts0   l0   w0  h0   x0  y0 z0 qx0 qy0 qz0 qw0 ts1 l1 w1 h1 x1 y1 z1 qx1
            # a = [4.75924, 2.02294, 1.55872, -26.214, 1.97311, 0.62196, -0.00570381, -0.000713125, 0.773904, 0.633276]
            rotation = Rotation.from_quat([-0.00570381, -0.000713125, 0.773904, 0.633276]).as_euler("XYZ").tolist()

            box9d = [[-26.214, 1.97311, 0.62196]+ [4.75924, 2.02294, 1.55872] + rotation]
            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera

            corners3d = transform_pts_with_T(corners, Tse_r @ Tew).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
        im = draw_boxes(im, corners2d)
        cv2.imwrite(f'/tmp/1234/5/{ts}.jpg', im)
        # import matplotlib.pyplot as plt
        # plt.imshow(im), plt.show()
