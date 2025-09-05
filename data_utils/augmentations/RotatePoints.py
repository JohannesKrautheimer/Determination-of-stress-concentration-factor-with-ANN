import torch
import numpy as np
import matplotlib.pyplot as plt

from data_utils.augmentations.PlotBeforeAndAfter import plot_before_and_after_lasernet, plot_before_and_after_pointnet

class RotatePoints(object):
    def __init__(self, max_degree, probability_application=0.1, model="pointnet2_regr_msg"):
        self.max_degree = max_degree
        self.probability_application = probability_application
        self.model = model
        
    def __call__(self, points):
        #Draw random number [0-1] and check if number is <= probability_application
        #If yes, move points in x direction
        random_number = torch.rand(1)[0] # [0, 1)
        if random_number <= self.probability_application:
            lower_bound = -self.max_degree
            upper_bound = self.max_degree
            random_rotation_degree = lower_bound + (upper_bound - lower_bound) * torch.rand(1).item()
            random_rotation_rad = np.radians(random_rotation_degree)
            if self.model == "pointnet2_regr_msg":
                #points have shape [num_points, 3]
                x = points[:, 0]
                y = points[:, 1]
                z = points[:, 2]

                rotated_x = x * np.cos(random_rotation_rad) - y * np.sin(random_rotation_rad)
                rotated_y = x * np.sin(random_rotation_rad) + y * np.cos(random_rotation_rad)
                rotated_points = np.column_stack((rotated_x, rotated_y, z))

                # Sort the points by their x-coordinate
                sorted_indices = np.argsort(rotated_points[:, 0])
                # old_points = np.copy(points)
                points = rotated_points[sorted_indices]

                # plot_before_and_after_pointnet(old_points, points, title="Rotate points")
                
            if self.model == "lasernet_regr":
                #points have shape [2, num_points]
                x = points[0, :]
                y = points[1, :]

                rotated_x = x * np.cos(random_rotation_rad) - y * np.sin(random_rotation_rad)
                rotated_y = x * np.sin(random_rotation_rad) + y * np.cos(random_rotation_rad)
                rotated_points = np.row_stack((rotated_x, rotated_y))

                # Sort the points by their x-coordinate
                sorted_indices = np.argsort(rotated_points[0, :])
                # old_points = np.copy(points)
                points = rotated_points[:, sorted_indices]

                # plot_before_and_after_lasernet(old_points, points, title="Rotate points")
        return points