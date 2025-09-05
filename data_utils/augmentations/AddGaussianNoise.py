import torch
import numpy as np

from data_utils.augmentations.PlotBeforeAndAfter import plot_before_and_after_lasernet, plot_before_and_after_pointnet

class AddGaussianNoiseToYCoordinate(object):
    def __init__(self, mean=0.0, std=1.0, model="pointnet2_regr_msg"):
        self.std = std
        self.mean = mean
        self.model = model
        
    def __call__(self, points):
        #Only add noise to y coordinate
        if self.model == "pointnet2_regr_msg":
            #points have shape [num_points, 3]
            num_points = len(points)
            noise_micro_meter = (torch.randn(num_points) * self.std + self.mean).numpy()
            noise_cm = noise_micro_meter / 10000
            # old_points = np.copy(points)
            points[:, 1] += noise_cm
            # plot_before_and_after_pointnet(old_points, points, title="Add noise")
        if self.model == "lasernet_regr":
            #points have shape [2, num_points]
            num_points = points.shape[1]
            noise_micro_meter = (torch.randn(num_points) * self.std + self.mean).numpy()
            noise_cm = noise_micro_meter / 10000
            # old_points = np.copy(points)
            points[1] += noise_cm
            # plot_before_and_after_lasernet(old_points, points, title="Add noise")
        return points