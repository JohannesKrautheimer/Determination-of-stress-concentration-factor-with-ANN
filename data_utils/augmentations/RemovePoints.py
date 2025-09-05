import torch
import numpy as np

from data_utils.augmentations.PlotBeforeAndAfter import plot_before_and_after_lasernet, plot_before_and_after_pointnet

#Points are not removed but rather set to the first value of the point cloud. Removing would result in problems because point clouds have all same size
class RemovePoints(object):
    def __init__(self, percentage_removed_points=0.3, probability_application=0.1, model="pointnet2_regr_msg"):
        self.percentage_removed_points = percentage_removed_points
        self.probability_application = probability_application
        self.model = model
        
    def __call__(self, points):
        #Draw random number [0-1] and check if number is <= probability_application
        #If yes, remove points
        random_number = torch.rand(1)[0] # [0, 1)
        if random_number <= self.probability_application:
            if self.model == "pointnet2_regr_msg":
                #points have shape [num_points, 3]
                N = points.shape[0]
                num_points_to_remove = int(N * self.percentage_removed_points)
                # Generate random indices for points to remove
                indices_to_remove = torch.randperm(N)[:num_points_to_remove]
                #Points are not removed but rather set to the first value of the point cloud.
                #Removing would result in problems because all point clouds need to have the same size
                # old_points = np.copy(points)
                points[indices_to_remove, :] = points[0, :]
                #sort points
                points = points[points[:, 0].argsort()]
                # plot_before_and_after_pointnet(old_points, points, title="Remove points")
            if self.model == "lasernet_regr":
                #points have shape [2, num_points]
                N = points.shape[1]
                num_points_to_remove = int(N * self.percentage_removed_points)
                # Generate random indices for points to remove
                indices_to_remove = torch.randperm(N)[:num_points_to_remove]
                #Points are not removed but rather set to the first value of the point cloud.
                #Removing would result in problems because all point clouds need to have the same size
                # old_points = np.copy(points)
                points[:, indices_to_remove] = points[:, 0][:, np.newaxis]
                #sort points
                sorted_indices = points[0].argsort()
                points = points[:, sorted_indices]
                # plot_before_and_after_lasernet(old_points, points, title="Remove points")
        return points