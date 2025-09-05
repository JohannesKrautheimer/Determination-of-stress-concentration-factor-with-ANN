from asyncio.log import logger
import os
from textwrap import indent
import numpy as np
import warnings
import pickle
import logging
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
from torch.utils.data import Dataset

# def plot_pc(point_cloud, label=""):
#     plt.figure()
#     plt.scatter(point_cloud[:, 0], point_cloud[:, 1], label=label)
#     plt.legend()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def sort_points_by_first_coordinate(data):
    # Sort the array based on the first column in ascending order
    # If values in the first column are equal, sort in descending order from the second column
    sorted_indices = np.lexsort((-data[:, 1], data[:, 0]))

    # Rearrange the rows of the array based on the sorted indices
    sorted_data = data[sorted_indices]
    return sorted_data

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """

    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class WeldScansPointnetDataSet(Dataset):
    def __init__(self, root, args, train_file=None, test_file=None, split='train', transforms=None):
        self.root = root
        self.npoints = args.num_point
        self.process_data = args.process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.transforms = transforms

        data_filenames = {}
        if train_file is not None:
            data_filenames['train'] = [line.rstrip() for line in open(train_file)]
        data_filenames['test'] = [line.rstrip() for line in open(test_file)]

        assert (split == 'train' or split == 'test')
        data_folders = ['_'.join(x.split('_')[0:-1]) for x in data_filenames[split]]
        self.datapath = [(data_folders[i], os.path.join(self.root, data_folders[i], data_filenames[split][i])) for i
                         in range(len(data_filenames[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'weldscans_%s_%dpts_fps.dat' % (split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'weldscans_%s_%dpts.dat' % (split, self.npoints))

        if self.process_data:
            # if not os.path.exists(self.save_path):
            print('Processing data (only running the first time)...')
            self.list_of_point_clouds = [None] * len(self.datapath)
            self.list_of_targets = [None] * len(self.datapath)

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                point_set, target = self.get_point_cloud_and_target(index)

                self.list_of_point_clouds[index] = point_set
                self.list_of_targets[index] = target

                # with open(self.save_path, 'wb') as f:
                #     pickle.dump([self.list_of_point_clouds, self.list_of_targets], f)
            # else:
            #     print('Load processed data from %s...' % self.save_path)
            #     with open(self.save_path, 'rb') as f:
            #         self.list_of_point_clouds, self.list_of_targets = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, target = self.list_of_point_clouds[index], self.list_of_targets[index]
        else:
            point_set, target = self.get_point_cloud_and_target(index)

        if self.transforms:
            for transform in self.transforms:
                point_set = transform(point_set)

        # plot_pc(point_set, label="before pc_normalize")
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, target[0]

    def __getitem__(self, index):
        return self._get_item(index)

    def get_point_cloud_and_target(self, index):
        fn = self.datapath[index]
        filename = fn[1]
        with open(filename) as f:
            line = f.readline()
            target = np.array([float(line.split(sep="=")[-1])]).astype(np.float32)
        point_set = np.loadtxt(filename, skiprows=1, delimiter=' ').astype(np.float32)
        #Set last coordinate to zero since otherwise this gives information which specimen or cut position it is
        point_set[:, 2] = 0

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
            point_set = sort_points_by_first_coordinate(point_set)
        else:
            point_set = point_set[0:self.npoints, :]
        return point_set, target

def get_max_sequence_len(lasernet_model, data_filenames, root):
    maximum_len = 0
    if lasernet_model.uniform:
        maximum_len = lasernet_model.npoints
    else:
        for split in ["train", "test"]:
            data_folders = ['_'.join(x.split('_')[0:-1]) for x in data_filenames[split]]
            datapath = [(data_folders[i], os.path.join(root, data_folders[i], data_filenames[split][i])) for i
                            in range(len(data_filenames[split]))]
            for dataset_entry in datapath:
                path = dataset_entry[1]
                with open(path) as f:
                    line = f.readline()
                    point_cloud = np.loadtxt(f, delimiter=' ', usecols=(0,1)).astype(np.float32)
                    if maximum_len < len(point_cloud):
                        maximum_len = len(point_cloud)
    return maximum_len

class WeldScansLasernetDataSet(Dataset):
    def __init__(self, root, args, train_file=None, test_file=None, split='train', max_sequence_len=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.uniform = args.use_uniform_sample
        self.npoints = args.num_point

        data_filenames = {}
        if train_file is not None:
            data_filenames['train'] = [line.rstrip() for line in open(train_file)]
        data_filenames['test'] = [line.rstrip() for line in open(test_file)]

        assert (split == 'train' or split == 'test')
        if max_sequence_len == None:
            self.max_len = get_max_sequence_len(self, data_filenames, self.root)
        else:
            self.max_len = max_sequence_len
        print(f"Max length Lasernet Split {split}: {self.max_len}")
        data_folders = ['_'.join(x.split('_')[0:-1]) for x in data_filenames[split]]
        self.datapath = [(data_folders[i], os.path.join(self.root, data_folders[i], data_filenames[split][i])) for i
                         in range(len(data_filenames[split]))]

        self.all_point_clouds = []
        self.all_targets = []
        for dataset_entry in self.datapath:
            path = dataset_entry[1]
            with open(path) as f:
                line = f.readline()
                target = np.array([float(line.split(sep="=")[-1])]).astype(np.float32)
                self.all_targets.append(target)
                point_cloud = np.loadtxt(f, delimiter=' ', usecols=(0,1)).astype(np.float32)
                # if self.max_len < len(point_cloud):
                #     self.max_len = len(point_cloud)
            if len(self.all_point_clouds) == 0:
                self.all_point_clouds = [point_cloud]
            else:
                self.all_point_clouds.append(point_cloud)
                
        for idx, point_set in enumerate(self.all_point_clouds):
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
                point_set = sort_points_by_first_coordinate(point_set)
            else:
                #Add padding to the data if uniform sample is not chosen
                num_pads = self.max_len - len(point_set)
                if num_pads > 0:
                    point_set = np.pad(point_set, ((0,num_pads),(0,0)), mode='constant', constant_values=0)
                if num_pads < 0:
                    #If length of point set is greater than max_len take away num_pad points from the end to get a length of max_len
                    point_set = point_set[:num_pads, :]
            point_set = np.transpose(point_set)
            self.all_point_clouds[idx] = point_set
        print('The number of %s samples is %d' % (split, len(self.all_point_clouds)))
        self.save_path = os.path.join(root, 'weldscans_%s.dat' % (split))

    def __len__(self):
        return len(self.all_point_clouds)

    def _get_item(self, index): 
        points = self.all_point_clouds[index]
        target = self.all_targets[index]
        
        if self.transforms:
            for transform in self.transforms:
                points = transform(points)
        return points, target

    def __getitem__(self, index):
        return self._get_item(index)

class WeldScansLasernetAlternativeDataSet(Dataset):
    def __init__(self, root, args, train_file=None, test_file=None, split='train'):
        self.root = root

        data_filenames = {}
        if train_file is not None:
            data_filenames['train'] = [line.rstrip() for line in open(train_file)]
        data_filenames['test'] = [line.rstrip() for line in open(test_file)]

        assert (split == 'train' or split == 'test')
        self.max_len = get_max_sequence_len(data_filenames, self.root)
        data_folders = ['_'.join(x.split('_')[0:-1]) for x in data_filenames[split]]
        self.datapath = [(data_folders[i], os.path.join(self.root, data_folders[i], data_filenames[split][i])) for i
                         in range(len(data_filenames[split]))]
        self.all_point_clouds = []
        self.all_targets = []
        for dataset_entry in self.datapath:
            path = dataset_entry[1]
            with open(path) as f:
                line = f.readline()
                target = np.array([float(line.split(sep="=")[-1])]).astype(np.float32)
                self.all_targets.append(target)
                point_cloud = np.loadtxt(f, delimiter=' ', usecols=(0,1)).astype(np.float32)
                # if self.max_len < len(point_cloud):
                #     self.max_len = len(point_cloud)
            if len(self.all_point_clouds) == 0:
                self.all_point_clouds = [point_cloud]
            else:
                self.all_point_clouds.append(point_cloud)
                
        #Add padding to the data
        for idx, point_set in enumerate(self.all_point_clouds):
            num_pads = self.max_len - len(point_set)
            point_set = np.pad(point_set, ((0,num_pads),(0,0)), mode='constant', constant_values=0)
            point_set = np.transpose(point_set)
            self.all_point_clouds[idx] = point_set
        print('The number of %s samples is %d' % (split, len(self.all_point_clouds)))

        # self.save_path = os.path.join(root, 'weldscans%s.dat' % (split))

    def __len__(self):
        return len(self.all_point_clouds)

    def _get_item(self, index): 
        points = self.all_point_clouds[index]
        target = self.all_targets[index]
        return points, target

    def __getitem__(self, index):
        return self._get_item(index)

if __name__ == '__main__':
    pass
    # import argparse

    # parser = argparse.ArgumentParser('loadData')
    # parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    # parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    # parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    # parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    # parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    # parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    # parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    # parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    # parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    # parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    # parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    # parser.add_argument('--neptune_offline', action='store_true', default=False, help='use offline mode for neptune ai')
    # args = parser.parse_args()
    # data = WeldScansDataLoader('data/weld_scans_2d_cuts/', args,split='train')