import numpy as np

def get_cameras(cali_file):
    with open(cali_file) as f:
        cali = yaml.safe_load(f)
    cameras = {}
    for cid in [cam]:
        cam_params = cali['rig'][cid]
        cameras[cid] = OpenCVFisheyeCamera(cam_params)
    return cameras





