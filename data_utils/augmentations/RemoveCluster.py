import numpy as np
import torch

from data_utils.augmentations.PlotBeforeAndAfter import plot_before_and_after_lasernet, plot_before_and_after_pointnet

class RemoveCluster(object):
    def __init__(self, n_points=40, probability_application=0.1, model="pointnet2_regr_msg"):
        self.n_points = n_points
        self.probability_application = probability_application
        self.model = model
        
    def __call__(self, points):
        #Draw random number [0-1] and check if number is <= probability_application
        #If yes, remove cluster
        random_number = torch.rand(1)[0] # [0, 1)
        if random_number <= self.probability_application:
            if self.model == "pointnet2_regr_msg":
                #points have shape [num_points, 3]
                N = points.shape[0]
                anchor_point = torch.randint(0, N, size=(1,)).numpy()[0]
                first_point = np.copy(points)[0, :]
                # Calculate the indices for the points to be set on the left and right side of the anchor point
                left_indices = np.arange(max(0, anchor_point - self.n_points // 2), anchor_point, dtype=np.int32)
                right_indices = np.arange(anchor_point, min(N, anchor_point + self.n_points // 2), dtype=np.int32)
                #Points are not removed but rather set to the first value of the point cloud.
                #Removing would result in problems because all point clouds need to have the same size
                # old_points = np.copy(points)
                points[left_indices, :] = first_point
                points[right_indices, :] = first_point
                #sort points
                points = points[points[:, 0].argsort()]
                # plot_before_and_after_pointnet(old_points, points, title="Remove cluster")
            if self.model == "lasernet_regr":
                #points have shape [2, num_points]
                N = points.shape[1]
                anchor_point = torch.randint(0, N, size=(1,)).numpy()[0]
                first_point = np.copy(points)[:, 0]
                # Calculate the indices for the points to be set on the left and right side of the anchor point
                left_indices = np.arange(max(0, anchor_point - self.n_points // 2), anchor_point, dtype=np.int32)
                right_indices = np.arange(anchor_point, min(N, anchor_point + self.n_points // 2), dtype=np.int32)
                #Points are not removed but rather set to the first value of the point cloud.
                #Removing would result in problems because all point clouds need to have the same size
                # old_points = np.copy(points)
                points[:, left_indices] = first_point[:, np.newaxis]
                points[:, right_indices] = first_point[:, np.newaxis]
                #sort points
                sorted_indices = points[0].argsort()
                points = points[:, sorted_indices]
                # plot_before_and_after_lasernet(old_points, points, title="Remove cluster")
        return points