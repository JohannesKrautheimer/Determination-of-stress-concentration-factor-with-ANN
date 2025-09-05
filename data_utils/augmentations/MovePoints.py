import torch
import numpy as np

from data_utils.augmentations.PlotBeforeAndAfter import plot_before_and_after_lasernet, plot_before_and_after_pointnet

class MovePointsXCoordinate(object):
    def __init__(self, max_movement_mm=5, probability_application=0.1, model="pointnet2_regr_msg"):
        self.max_movement_mm = max_movement_mm
        self.probability_application = probability_application
        self.model = model
        
    def __call__(self, points):
        #Draw random number [0-1] and check if number is <= probability_application
        #If yes, move points in x direction
        random_number = torch.rand(1)[0] # [0, 1)
        if random_number <= self.probability_application:
            # Generate a random number between [-max_movement, +max_movement]
            lower_bound = -self.max_movement_mm
            upper_bound = self.max_movement_mm
            random_movement_x_mm = lower_bound + (upper_bound - lower_bound) * torch.rand(1).item()
            random_movement_x_cm = random_movement_x_mm / 10
            # old_points = np.copy(points)
            if self.model == "pointnet2_regr_msg":
                #points have shape [num_points, 3]
                points[:, 0] = points[:, 0] + random_movement_x_cm
                # plot_before_and_after_pointnet(old_points, points, "Move points")
            if self.model == "lasernet_regr":
                #points have shape [2, num_points]
                points[0, :] = points[0, :] + random_movement_x_cm
                # plot_before_and_after_lasernet(old_points, points, "Move points")
            
        return points