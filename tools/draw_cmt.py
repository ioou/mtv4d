from pathlib import Path as P
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mtv4d.utils.geo_base import transform_pts_with_T, Rt2T
from mtv4d.utils.box_base import fbbox_to_box9d, jsonbox_to_box9d
from mtv4d.utils.draw_base import draw_boxes
from mtv4d.annos_4d.misc import read_ego_paths
from mtv4d.utils.box_base import to_corners_9
from mtv4d.utils.io_base import read_json, read_pickle
from mtv4d.utils.sensors import get_camera_models, FisheyeCameraModel
from mtv4d.utils.calib_base import read_cal_data
import cv2
import argparse
import os.path as op

parser = argparse.ArgumentParser()
parser.add_argument("--mode")


def main():
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
        elif False:  # output right-front-up
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

            box9d = [[-26.214, 1.97311, 0.62196] + [4.75924, 2.02294, 1.55872] + rotation]
            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera

            corners3d = transform_pts_with_T(corners, Tse_r @ Tew).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
        im = draw_boxes(im, corners2d)
        cv2.imwrite(f'/tmp/1234/5/{ts}.jpg', im)
        # import matplotlib.pyplot as plt
        # plt.imshow(im), plt.show()


def fbbox_filter_dist_class(box, class_list=None, dist_thres=None):
    # default not filter
    if class_list is not None and box['class'] not in class_list:
        return False
    if dist_thres is not None and max(box['translation']) < dist_thres:
        return False
    return True


def draw_pickle_cmt():
    path = "val_indice.pkl"


# todo: mp_pool -> 30s; cat 4 images to one image
def draw_pickle_fb(pickle_path, data_root, draw_class_list=None, THRES_dist_to_ego=None):
    sensors = ["camera1", "camera5", "camera8", "camera11"]
    sensor_id = sensors[2]
    a = read_pickle(pickle_path)
    for scene_id, infos in tqdm(a.items()):
        calib = infos['scene_info']['calibration']
        T_es = Rt2T(calib[sensor_id]['extrinsic'][0], calib[sensor_id]['extrinsic'][1])
        T_se = np.linalg.inv(T_es)
        camera_model = FisheyeCameraModel.from_intrinsic_array(calib[sensor_id]['intrinsic'], sensor_id)
        for ts, data in infos['frame_info'].items():
            box_info, poly_info = data['3d_boxes'], data['3d_polylines']
            box_filtered = [box for box in box_info if fbbox_filter_dist_class(box, draw_class_list, THRES_dist_to_ego)]
            box9d, box_class_names = [fbbox_to_box9d(box) for box in box_filtered], [box['class'] for box in
                                                                                     box_filtered]
            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera
            corners3d = transform_pts_with_T(corners, T_se).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
            im = cv2.imread(op.join(data_root, data['camera_image'][sensor_id]))
            im = draw_boxes(im, corners2d)
            if False:  # draw label
                for idx, (box, text) in enumerate(zip(corners2d.reshape(-1, 8), box_class_names)):
                    cv2.putText(im, f'{idx}_{text.split(".")[-1]}', tuple(box[:2].astype('int')), 1, 1, (255, 0, 0))
            save_path = f"/tmp/1234/fb/{sensor_id}_{scene_id}_{ts}.jpg"
            P(save_path).parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(save_path, im)

def draw_pickle_fb_ps(pickle_path, data_root, draw_class_list=None, THRES_dist_to_ego=None):
    sensors = ["camera1", "camera5", "camera8", "camera11"]
    sensor_id = sensors[2]
    a = read_pickle(pickle_path)
    for scene_id, infos in tqdm(a.items()):
        calib = infos['scene_info']['calibration']
        T_es = Rt2T(calib[sensor_id]['extrinsic'][0], calib[sensor_id]['extrinsic'][1])
        T_se = np.linalg.inv(T_es)
        camera_model = FisheyeCameraModel.from_intrinsic_array(calib[sensor_id]['intrinsic'], sensor_id)
        for ts, data in infos['frame_info'].items():
            box_info, poly_info = data['3d_boxes'], data['3d_polylines']
            # box_filtered = [box for box in box_info if fbbox_filter_dist_class(box, draw_class_list, THRES_dist_to_ego)]
            # box9d, box_class_names = [fbbox_to_box9d(box) for box in box_filtered], [box['class'] for box in
            #                                                                          box_filtered]
            corners = np.array([i['points'] for i in poly_info if 'slot' in i['class']])  # ego_base -> camera
            corners3d = transform_pts_with_T(corners, T_se).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
            im = cv2.imread(op.join(data_root, data['camera_image'][sensor_id]))
            for i in corners2d.reshape(-1, 4, 2):
                cv2.polylines(im, [i.astype('int')], 1, (0, 0, 255), 1)
                cv2.polylines(im, [i.astype('int')[:2]], 1, (0, 0, 255), 1)
            # im = draw_boxes(im, corners2d)
            if False:  # draw label
                for idx, (box, text) in enumerate(zip(corners2d.reshape(-1, 8), box_class_names)):
                    cv2.putText(im, f'{idx}_{text.split(".")[-1]}', tuple(box[:2].astype('int')), 1, 1, (255, 0, 0))
            save_path = f"/tmp/1234/fb/{sensor_id}_{scene_id}_{ts}.jpg"
            P(save_path).parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(save_path, im)


# todo: mp_pool -> 30s; cat 4 images to one image
def draw_pickle_lfcjson_with_json(json_dir, data_root, draw_class_list=None, THRES_dist_to_ego=None):
    sensors = ["camera1", "camera5", "camera8", "camera11"]
    sensor_id = sensors[2]
    a = load_points_from_jsons(json_dir)
    for scene_id, infos in tqdm(a.items()):
        calib = infos['scene_info']['calibration']
        T_es = Rt2T(calib[sensor_id]['extrinsic'][0], calib[sensor_id]['extrinsic'][1])
        T_se = np.linalg.inv(T_es)
        camera_model = FisheyeCameraModel.from_intrinsic_array(calib[sensor_id]['intrinsic'], sensor_id)
        for ts, data in infos['frame_info'].items():
            box_info, poly_info = data['3d_boxes'], data['3d_polylines']
            box_filtered = [box for box in box_info if fbbox_filter_dist_class(box, draw_class_list, THRES_dist_to_ego)]
            box9d, box_class_names = [fbbox_to_box9d(box) for box in box_filtered], [box['class'] for box in
                                                                                     box_filtered]
            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera
            corners3d = transform_pts_with_T(corners, T_se).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
            im = cv2.imread(op.join(data_root, data['camera_image'][sensor_id]))
            im = draw_boxes(im, corners2d)
            if False:  # draw label
                for idx, (box, text) in enumerate(zip(corners2d.reshape(-1, 8), box_class_names)):
                    cv2.putText(im, f'{idx}_{text.split(".")[-1]}', tuple(box[:2].astype('int')), 1, 1, (255, 0, 0))
            save_path = f"/tmp/1234/fb/{sensor_id}_{scene_id}_{ts}.jpg"
            P(save_path).parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(save_path, im)


def is_box(obj_dict):
    return obj_dict['geometry_type'] == 'box3d'


def is_poly(obj_dict):
    return obj_dict['geometry_type'] == 'polyline3d'


def draw_pickle_4djson(json_path, data_root, draw_class_list=None):
    scene_id = P(json_path).parent.parent.name
    scene_root = op.join(data_root, scene_id)
    sensors = ["camera1", "camera5", "camera8", "camera11"]
    sensor_id = sensors[2]
    data = read_json(json_path)
    timestamps = [i['timestamp'] for i in data[0]['ts_list_of_dict']]
    data_boxes = [box for box in data if is_box(box) and fbbox_filter_dist_class(box, draw_class_list, None)]
    # a = data_boxes[218]['ts_list_of_dict'][50]['visibility']
    boxes_, box_labels_ = [box['geometry'] for box in data_boxes], [box['obj_type'] for box in data_boxes]
    data_polylines = [poly for poly in data if is_poly(poly)]
    polys_, poly_labels_ = [poly['geometry'] for poly in data_polylines], [poly['geometry'] for poly in data_polylines]
    Twes, _ = read_ego_paths(P(scene_root)/'trajectory.txt')
    calib_path = str(P(scene_root)/'calibration_center.yml')
    calib = read_cal_data(calib_path)
    camera_model = get_camera_models(calib_path)[sensor_id]

    def get_box_visibility(box_dict, ts, sensor_id):
        for i in box_dict:
            if int(i['timestamp']) == int(ts):
                return i['visibility'][sensor_id]

    def is_visible(box_dict, ts, sensor_id):
        # print(get_box_visibility(box_dict, ts, sensor_id), get_box_visibility(box_dict, ts, 'lidar1'))
        return max(get_box_visibility(box_dict, ts, sensor_id), get_box_visibility(box_dict, ts, 'lidar1')) > 0

    for idx, ts in enumerate(timestamps):
        visible_box_mask = [ is_visible(box['ts_list_of_dict'], ts, sensor_id) for box in data_boxes ]
        boxes, box_labels = [i for i, j in zip(boxes_, visible_box_mask) if j], [i for i, j in zip(box_labels_, visible_box_mask) if j]
        T_sw = calib[sensor_id]['T_se'] @ np.linalg.inv(Twes[float(ts)])
        im = cv2.imread(op.join(scene_root, f'camera/{sensor_id}/{int(ts)}.jpg'))
        if False:
            box9d, box_class_names = [jsonbox_to_box9d(box) for box in boxes], box_labels
            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera
            corners3d = transform_pts_with_T(corners, T_sw).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
            im = draw_boxes(im, corners2d)
            if True:  # draw label
                # text_list = box_class_names = [text.split(".")[-1] for text in box_class_names]
                text_list = [str(get_box_visibility(box['ts_list_of_dict'], ts, sensor_id)) for box in data_boxes]
                text_list = [i for i, j in zip(text_list, visible_box_mask) if j]
                for idx, (box, text) in enumerate(zip(corners2d.reshape(-1, 8, 2), text_list)):
                    cv2.putText(im, f'{idx}_{text}', tuple(box[0].astype('int')), 1, 2, (255, 0, 0), 2)
        if True:
            polys, poly_classes_names = [np.array(poly) for poly in polys_], poly_labels_
            visibility = ''.join([vis['ts_list_of_dict'][0]['visibility'][sensor_id] for vis in data_polylines])
            corners3d = [ transform_pts_with_T(i, T_sw) for i in polys]
            corners2d = [camera_model.project_points(pts) for pts in corners3d]
            num = 0
            for i in corners2d:
                cv2.polylines(im, [i.reshape(-1,2).astype('int')], 0, (0, 0, 255))
                for j in i:
                    clr = (0, 255, 0) if visibility[num] == '1' else (255, 0, 0)
                    cv2.circle(im, (int(j[0]), int(j[1])), 1, clr, 2)
                    num+=1
        save_path = f"/tmp/1234/4djson/{sensor_id}_{scene_id}_{ts}.jpg"
        P(save_path).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(save_path, im)


if __name__ == "__main__":
    # main()
    if False:  # fb pickle
        # pickle_path = "/ssd1/MV4D_12V3L/20250315_153742_1742024382664_1742024467964/fb_fix_20250315_153742_1742024382664_1742024467964.pkl"
        pickle_path = "/ssd1/MV4D_12V3L/20231107_163857/fb_fix_20231107_163857.pkl"
        data_root = '/ssd1/MV4D_12V3L'
        draw_pickle_fb(pickle_path, data_root)
    if False:  # fb pickle, ps
        # pickle_path = "/ssd1/MV4D_12V3L/20250315_153742_1742024382664_1742024467964/fb_fix_20250315_153742_1742024382664_1742024467964.pkl"
        pickle_path = "/ssd1/MV4D_12V3L/20231107_163857/fb_fix_20231107_163857.pkl"
        data_root = '/ssd1/MV4D_12V3L'
        draw_pickle_fb_ps(pickle_path, data_root)
    if False:  # 4djson
        json_path = "/ssd1/tmp/20231104_170321_1699088601564_1699088721564/4d_anno_infos/annos.json"
        data_root = "/ssd1/tmp/"
        draw_pickle_4djson(json_path, data_root)

    # if True:  # libo json
    #     json_path = "/home/yuanshiwei/4/prefusion/work_dirs/borui_dets_71/gt_pred_dumps/dets"
    #     data_root = '/ssd1/MV4D_12V3L'
    #     draw_pickle_lfcjson(json_path, data_root)

    if False:  # frame json, not implemented
        json_path = '/ssd1/tmp/20231108_170321'
        data_root = '/ssd1/tmp'
        draw_pickle_frame_json(json_path, data_root)

    if True:  # 4djson
        json_path = "/home/yuanshiwei/data/3dgs_demo3/4d_anno_infos/annos.json"
        data_root = '/home/yuanshiwei/data'
        draw_pickle_4djson(json_path, data_root)