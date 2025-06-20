from scipy.stats import multivariate_normal
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt
from dsnt import dsnt


def build_heatmap(target_pos) :
    '''
    For a list of target keypoints, generate one gaussian heatmap per keypoint
    '''
    target_keypoints = target_pos.reshape(3,-1)
    keypoints_heatmaps = np.array((8,56,56))
    pos = np.dstack(np.mgrid[0:56:1, 0:56:1])
    
    
    for i, keypoint in enumerate(target_keypoints):
        rv = multivariate_normal(mean=[keypoint[0], keypoint[1]], cov=4)
        if keypoint[2] : #if the keypoint is visible
            keypoints_heatmaps[i] = rv.pdf(pos)
    
    return tf.convert_to_tensor(keypoints_heatmaps)

def combined_loss(targets,pred):
    
    #target_pose = tf.reshape(targets, [pred.shape[0],2,pred.shape[-1]])
    sum_loss = 0
    for ch in range(pred.shape[-1]):
        norm_heatmaps, coords = dsnt.dsnt(pred[:,:,:,ch])
        sum_loss+= tf.losses.mean_squared_error(coords,targets[:,:,ch]) +  dsnt.js_reg_loss(norm_heatmaps, targets[:,:,ch], fwhm=3)
    print("sum_loss :", tf.math.reduce_sum(sum_loss))
    return tf.math.reduce_sum(sum_loss)


def mse_loss(target_pos, pred_pos):
    keypoints_heatmaps = build_heatmap(target_pos)
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(keypoints_heatmaps, pred_pos)
    return mse_loss

def mse_loss(target_pos, pred_pos):
    keypoints_heatmaps = build_heatmap(target_pos)
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(keypoints_heatmaps, pred_pos)
    return mse_loss


def focal_loss(target_pos, pred_pos):
    keypoints_heatmaps = build_heatmap(target_pos)
