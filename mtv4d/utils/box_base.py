import numpy as np
from scipy.spatial.transform import Rotation
import sys

sys.path.append(".")
import copy

"""
所有的bbx的形式
前5个都是前左上
1 collaborator
2 psr
3 4djson
4 7points
5 9points
6 others: pcddet box 
"""


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
    R = Rotation.from_euler("XYZ", [u, v, w]).as_matrix()
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
    pts = np.array(pts)
    x, y, z = pts[:3]
    a, b, c = pts[3:6] / 2
    u, v, w = pts[6], pts[7], pts[8]
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

    R = Rotation.from_euler("XYZ", [u, v, w]).as_matrix()
    corners = (R @ corners.T).T
    corners += np.array([x, y, z]).reshape(-1, 3)
    return corners


def anno_box_to_9_values_box(box: dict) -> np.ndarray:
    return np.array(
        [
            box["psr"]["position"]["x"],
            box["psr"]["position"]["y"],
            box["psr"]["position"]["z"],
            box["psr"]["scale"]["x"],
            box["psr"]["scale"]["y"],
            box["psr"]["scale"]["z"],
            box["psr"]["rotation"]["x"],
            box["psr"]["rotation"]["y"],
            box["psr"]["rotation"]["z"],
        ],
        dtype=np.float32,
    )


def transform_output_geometry_to_array(box):
    return [
        *box["pos_xyz"],
        *box["scale_xyz"],
        *box["rot_xyz"],
    ]


def transform_psr_to_array(psr):
    return [
        psr["position"]["x"],
        psr["position"]["y"],
        psr["position"]["z"],
        psr["scale"]["x"],
        psr["scale"]["y"],
        psr["scale"]["z"],
        psr["rotation"]["x"],
        psr["rotation"]["y"],
        psr["rotation"]["z"],
    ]


def translate_psr_to_output_geometry(psr):
    output = {}
    output["pos_xyz"] = [psr["position"]["x"], psr["position"]["y"], psr["position"]["z"]]
    output["scale_xyz"] = [psr["scale"]["x"], psr["scale"]["y"], psr["scale"]["z"]]
    output["rot_xyz"] = [psr["rotation"]["x"], psr["rotation"]["y"], psr["rotation"]["z"]]
    return output


def translate_output_geometry_to_psr(box):
    output = {
        "position": {
            "x": box["pos_xyz"][0],
            "y": box["pos_xyz"][1],
            "z": box["pos_xyz"][2],
        },
        "scale": {
            "x": box["scale_xyz"][0],
            "y": box["scale_xyz"][1],
            "z": box["scale_xyz"][2],
        },
        "rotation": {
            "x": box["rot_xyz"][0],
            "y": box["rot_xyz"][1],
            "z": box["rot_xyz"][2],
        },
    }
    return output


def to_list_psr_box_from_vec7_conf(boxes, confs, labels):
    output = []
    for b, c, l in zip(boxes, confs, labels):
        output.append({"obj_attr": {}, "obj_type": l, "psr": box_vec7_to_psr(b), "conf": c})
    return output


def box_vec7_to_psr(vec7):
    return {
        "position": {
            "x": vec7[0],
            "y": vec7[1],
            "z": vec7[2],
        },
        "scale": {
            "x": vec7[3],
            "y": vec7[4],
            "z": vec7[5],
        },
        "rotation": {"x": 0, "y": 0, "z": vec7[6]},
    }
def frame_box_to_psr(box):
    return {
        "position": {
            "x": box['pos_xyz'][0],
            "y": box['pos_xyz'][1],
            "z": box['pos_xyz'][2],
        },
        "scale": {
            "x":  box['scale_xyz'][0],
            "y":  box['scale_xyz'][1],
            "z":  box['scale_xyz'][2],
        },
        "rotation": {
            "x":  box['rot_xyz'][0],
            "y":  box['rot_xyz'][1],
            "z":  box['rot_xyz'][2],
        },
    }


def translate_psr_with_T(psr, Tts):
    # Tts 表示 从box的当前系转到目标系
    position = np.array([[psr["position"]["x"], psr["position"]["y"], psr["position"]["z"], 1.0]])
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_euler("XYZ", [psr["rotation"]["x"], psr["rotation"]["y"],
                                                   psr["rotation"]["z"]]).as_matrix()
    transformed_position = (position @ Tts.T).squeeze()
    transformed_rotation = Rotation.from_matrix((Tts @ rotation)[:3, :3]).as_euler("XYZ")
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

def box_corners_to_dot_cloud(corners):
    # 保证是3维度的点
    assert np.array(corners).shape[-1] == 3
    def get_lines(box):
        output = []
        for i, j in [
            [box[0], box[1]],
            [box[1], box[2]],
            [box[2], box[3]],
            [box[3], box[0]],
            [box[0 + 4], box[1 + 4]],
            [box[1 + 4], box[2 + 4]],
            [box[2 + 4], box[3 + 4]],
            [box[3 + 4], box[0 + 4]],
            [box[0], box[0 + 4]],
            [box[1], box[1 + 4]],
            [box[2], box[2 + 4]],
            [box[3], box[3 + 4]],
        ]:
            for n in range(100):
                n = n/100
                output += [i *(1-n) + j *n]
        return output
    box_points = np.array([get_lines(i) for i in corners.reshape(-1, 8, 3)]).reshape(-1, 3)
    return box_points
