from visualize_data import Visualizer
import os
import sys
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

root_frames_dir = "/home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes/generated_dataset/test"
model_path = "/home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes/model_20241203_1634_.keras"
K_path = "/home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes/DataVisualization/camera.json"
dest_dir = "/home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes/visualizations"

visualizer = Visualizer(root_frames_dir, model_path, K_path, dest_dir)

def calcalate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

memo = []
#take random files in that dir
for filename in os.listdir(os.path.join(root_frames_dir, "seq_RT002")):
    if filename.endswith(".png"):
        if visualizer.create_graphic(filename) is not None:
            q_label, r_label, q_pred, r_pred = visualizer.create_graphic(filename)
        memo.append((q_label, r_label, q_pred, r_pred))