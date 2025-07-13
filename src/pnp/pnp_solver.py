import cv2
import numpy as np
import json
import os
from omegaconf import OmegaConf

from scipy.spatial.transform import Rotation as R

class PnPSolver:
    def __init__(self):
        config = OmegaConf.load("/Users/jost/Jost/Code/2024-nc-hackathon-spades/configs/mobilenet.yaml")
        with open(config.root.labels + "/camera.json", "r") as f:
            data = json.load(f)
            self.cam_matrix = np.array(data["cameraMatrix"], dtype=float)
            self.dist_coeffs = np.array(data["distCoeffs"], dtype=float)

        with open(config.root.labels + "/points.json", "r") as f:
            data = json.load(f)
            self.scaling = data["mockup_scale_ratio"]
            self.object_points = np.array(data["keypoints"], dtype=float)

    def solve_pnp(self, image_points):
        success, rvec, tvec, inliers = cv2.solvePnPRansac(self.object_points, image_points, self.cam_matrix, self.dist_coeffs)
        rmat, _ = cv2.Rodrigues(rvec)
        rot_m = R.from_matrix(rmat)
        q = rot_m.as_quat()  # (x, y, z, w) format.
        r = tvec.reshape(3,) * self.scaling
        return r, q
    

