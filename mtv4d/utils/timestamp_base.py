from pathlib import Path as P
import os.path as op
from mtv4d.annos_4d.helper import get_sync_filename, get_times, read_json_to_list, write_json_from_list
import numpy as np

cameras = [
    "camera1",
    "camera5",
    "camera8",
    "camera11",
    "camera2",
    "camera3",
    "camera4",
    "camera7",
    "camera6",
    "camera15",
    "camera12",
]


class Timestamps:
    def __init__(self, scene_root, cameras, data=None):
        self.scene_root = scene_root
        self.cameras = cameras
        if data is None:
            self.generate_time_json_with_none(scene_root, cameras)
        else:
            self.timestamp_list = data

    @classmethod
    def from_json(cls, scene_root, cameras):
        path = op.join(scene_root, "4d_anno_infos/ts_full.json")
        data = read_json_to_list(path)
        return cls(scene_root, cameras, data)

    def generate_time_json_with_none(self, scene_root, cameras):
        timestamps_traj = np.loadtxt(P(scene_root) / "whole_scene/ego-trajectory/trajectory_temp_horizontal.txt")
        lidar_path = P(scene_root) / "lidar/undistort_static_lidar1"
        timestamps, _ = get_times(lidar_path)
        output_ts_list = [{"lidar": ts} for ts in timestamps if ts in timestamps_traj]

        # overlapped_lidar1
        new_lidar1 = P(scene_root) / "lidar/overlapped_lidar1"
        new_lidar1 = new_lidar1 if new_lidar1.exists() else P(scene_root) / f"hidden_point_removal/occlusion_filter_lidar_lidar1"
        label_times, label_names = get_times(new_lidar1)
        [i.update({'overlapped_lidar1': get_sync_filename(label_names, label_times, i["lidar"])}) for i in output_ts_list]


        for cam in cameras:
            camera_path = P(scene_root) / "camera" / cam
            label_times, label_names = get_times(camera_path)
            [i.update({cam: get_sync_filename(label_names, label_times, i["lidar"])}) for i in output_ts_list]

            # camera_hpr_path = P(scene_root) / f"hidden_point_removal/occlusion_filter_lidar_{cam}"
            # label_times, label_names = get_times(camera_hpr_path)
            # [i.update({f"hpr_{cam}": get_sync_filename(label_names, label_times, i["lidar"])}) for i in output_ts_list]
        write_json_from_list(output_ts_list, op.join(scene_root, "4d_anno_infos/ts_full.json"))
        self.timestamp_list = output_ts_list
        

    @property
    def timestamps_dict(self):
        return {i["lidar"]: i for i in self.timestamp_list}

    @property
    def timestamps_dict_clean(self):
        return {i["lidar"]: i for i in self.timestamp_list if all(i.values())}

    @property
    def timestamps_given_cameras(self, cameras):
        return {i["lidar"]: [i for i in cameras] for i in self.timestamp_list if all(i.values())}

    @property
    def timestamps_full(self):
        return [i["lidar"] for i in self.timestamp_list]

    @property
    def timestamps_clean(self):
        print( len([i["lidar"] for i in self.timestamp_list if all(i.values())]))
        return [i["lidar"] for i in self.timestamp_list if all(i.values())]


if __name__ == "__main__":
    from time import time

    a = time()
    ts = Timestamps("/ssd1/data/4d/20230823_110018", cameras)
    print(time() - a)
    print(len(ts.timestamps_dict), len(ts.timestamps_dict_clean))
    a = time()
    ts = Timestamps.from_json("/ssd1/data/4d/20230823_110018", cameras)
    print(len(ts.timestamps_dict), len(ts.timestamps_dict_clean))
    print(time() - a)
