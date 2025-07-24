import json
import matplotlib.pyplot as plt
# creat a 3d diagram of the keypoints
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import pandas as pd
import logging

import os
import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')
from datetime import datetime
import pandas as pd
import glob
import logging
import sys

from tensorflow.keras.callbacks import ModelCheckpoint

from src.dataloaders.spades import create_dataset
from src.losses.poseloss import geodesic_dist, position_mse_loss, ori_error, rel_l2_error, mpkpe_heatmap, mpkpe_regression
from src.losses.heatmaploss import heatmap_loss
from src.models.mobilenet_regression import MobilenetModel_regression
from src.models.mobilenet_heatmap import mobilenet_heatmap
import keras
from src.utils import dsnt



logging.basicConfig(stream=sys.stderr, level=logging.INFO)

write = True
data_path = "data/bbox/keypoints.csv"
model_name = "networks/model_arun_heatmaps.keras"
input_size = (224,224,3)
display = True

model = keras.models.load_model(model_name)
model.summary()


data_dict = pd.read_csv(data_path).to_dict()
item_list = {'filepath':[], 'bbox':[]}
for i in range(8) :
    item_list[f'k{i}x']=[]
    item_list[f'k{i}y']=[]
            
for img_idx in range(len(data_dict["filepath"])):
    path = "/home/lecomte/Documents/2024-nc-hackathon-AstroSpikes/data/bbox/" + data_dict["filepath"][img_idx].split("/")[-1]
    if os.path.isfile(path):
        for key in item_list.keys():
            if key=='filepath' :
                item_list[key].append(path)
            else :
                item_list[key].append(data_dict[key][img_idx])

print(len(item_list['filepath']))
print(item_list['filepath'][0])

train_dataset = create_dataset(item_list,
                                batch_size=1,
                                input_size=input_size[:2],
                                is_training=False,
                                cache_dir=None)

print(len(train_dataset))

dummy_input, dummy_target = next(iter(train_dataset))
print(dummy_input.shape)
resolution = (dummy_input.shape[1], dummy_input.shape[2])



display_resolution = (resolution[0], resolution[1])
reg_video = cv2.VideoWriter("Regressed keypoints.avi", 0, 3.0, display_resolution,1)
cv2.namedWindow("Combined regression", cv2.WINDOW_NORMAL )

MPJPE_accumulated = 0
current_file = ""

for batch_idx, (data, target) in enumerate(train_dataset):
    
    output = model.predict(data, verbose=0)
    #initialize event image            
    detection_threshold = 0.1
    target = target
    output = output
    #predictions
    #2 dimensions per head for voltage and current

    # print(output.shape) #torch.Size([1,56,56,8])
    #pred_coords = np.reshape(output,(8,2))
        # 1. Ensure y_pred_heatmaps is in channels_first format (B, C, H, W) for dsnt
    if tf.rank(output) == 4:
        pred_shape = tf.shape(output)
        # Heuristic for (B, H, W, C) -> Transpose to (B, C, H, W)
        if pred_shape[1] > pred_shape[3]:
            y_pred_heatmaps = tf.transpose(output, perm=[0, 3, 1, 2])

    # 2. Convert predicted heatmaps to PIXEL coordinates.
    #    The output `y_pred_coords` will have shape (Batch, Locations, 2).
    y_pred_coords = dsnt.dsnt(y_pred_heatmaps, normalized_coordinates=False, method='softmax')[0]

    cx_pr, cy_pr = np.mean(y_pred_coords, axis=0, dtype=np.uint16)

    #ground truths
    #print(target.shape) #torch.Size([1, 8, 2])
    gt_coords = np.reshape(target,(8,2))
    cx_gt, cy_gt = np.mean(gt_coords, axis=0, dtype=np.uint16)
    all_image = data[0].numpy()
    coords = np.empty((8,), dtype=np.dtype([('x', '<u2'), ('y', '<u2')]))
    all_image_write = (all_image*255).astype(np.uint16)
    MPJPE = 0

    for joint in range(8):

        k_x, k_y = y_pred_coords[joint]
        gt_x, gt_y = gt_coords[joint]
        cv2.circle(all_image, (int(k_x*4), int(k_y*4)), 4, (0, 0, 200), thickness=-1)
        cv2.circle(all_image, (int(gt_x*112+112), int(gt_y*112+112)), 2, (200, 200, 200), thickness=-1)

        JPE = np.linalg.norm((k_x - gt_x, k_y - gt_y))
        MPJPE += JPE

    all_image_write = (all_image*255).astype(np.uint16)  
    if write :
        reg_video.write(all_image_write)
        cv2.waitKey(delay=1)
    if display :
        cv2.imshow("Combined regression", all_image) #imshow display (y,x,#channels) when given a (x,y,#channels) array
        cv2.waitKey(delay=300)

logging.info("MJPE : " + str(MPJPE_accumulated/(len(train_dataset))))
logging.info("Releasing the writers")        
reg_video.release()

cv2.destroyAllWindows()