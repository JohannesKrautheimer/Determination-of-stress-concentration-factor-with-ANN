"""
Author: Öner Aydogan
Based on: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import os
import copy
import sys
import torch
import numpy as np
import datetime
import logging
from data_utils.WeldScansDataLoader import WeldScansLasernetDataSet, WeldScansPointnetDataSet
from data_utils.augmentations.AddGaussianNoise import AddGaussianNoiseToYCoordinate
from data_utils.augmentations.MovePoints import MovePointsXCoordinate
from data_utils.augmentations.RemoveCluster import RemoveCluster
from data_utils.augmentations.RemovePoints import RemovePoints
import shutil
import argparse
import neptune.new as neptune

from pathlib import Path
from Trainer import Trainer, inplace_relu
from data_utils.augmentations.RotatePoints import RotatePoints

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode (default: false)')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device (default: 0)')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training (default: 24)')
    parser.add_argument('--model', default='pointnet2_regr_msg', help='model name (default: pointnet2_regr_msg)')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training (default: 100)')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training (default: 0.001)')
    parser.add_argument('--num_point', type=int, default=256, help='Point Number (default: 256)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training (default: Adam')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root (default: None)')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate (default: 0.0001)')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals (default: False)')
    parser.add_argument('--process_data', action='store_true', default=False, help='process data before run instead of loading it on each iteration (default: false)')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling (default: false)')
    parser.add_argument('--neptune_mode', choices=['sync', 'offline', 'debug'], default='offline', help='mode for neptune ai (sync, offline or debug). Default: offline')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--loss_fn', choices=["mae", "mse", "r2"], default='r2', help='loss function for training (mae, mse or r2). Default: r2')
    parser.add_argument('--b1', type=float, default=0.9, help='beta 1 for Adam Optimizer (default 0.9)')
    parser.add_argument('--tag', type=str, default='', help='neptune tag for the run')
    parser.add_argument('--data_path', type=str, default='data/weld_scans_2d_cuts_rotated_in_cm/', help='path to the location of the data (default: data/weld_scans_2d_cuts_rotated_in_cm/)')
    parser.add_argument('--set_path', type=str, default='train_test_sets/weld_scans_2d_cuts/default/', help='path to the directory with the train and test set configs (default: train_test_sets/weld_scans_2d_cuts/default/)')
    parser.add_argument('--move_points', type=float, default=None, nargs=2, help='move all points from the point cloud in x direction with the given maximum movement in mm & probabilty of application [0-1]. The point cloud is randomly moved between (-max_movement, +max_movement)')
    parser.add_argument('--rotate_points', type=float, default=None, nargs=2, help='rotate points from the point cloud with the given maximum angle in degrees & probabilty of application [0-1]. The point cloud is randomly rotated between (-max_degree, +max_degree)')
    parser.add_argument('--add_noise', type=float, default=None, help='add gaussian noise to the data with the given standard deviation in mikrometer(mean = 0)')
    parser.add_argument('--remove_points', type=float, default=None, nargs=2, help='remove points from the point cloud with the given percentage of removed points [0-1] & probabilty of application [0-1]')
    parser.add_argument('--remove_cluster', type=float, default=None, nargs=2, help='remove a cluster from the point cloud with the given number of points & probability of application [0-1]')
    parser.add_argument('--cv', type=int, default=None, help='Apply k-fold cross validation with the given int as the number of folds')

    parsed_args = parser.parse_args()
    if "SLURM_JOB_NAME" in os.environ:
        slurm_job_name = os.getenv("SLURM_JOB_NAME")
        parsed_args.tag = slurm_job_name
    return parsed_args

def main(args):
    
    run = neptune.init(
        project="aydo96/spannungskonzentration",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiOTYxOWI1Ni0yZmI2LTQ2ZDEtYTFkNC1lNmU5NjM3ODliYmQifQ==",
        mode=args.neptune_mode
    )

    if args.tag != "":
        run['sys/tags'].add([args.tag])

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('regression')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        str1 = "" if args.tag == "" else (args.tag + "_")
        exp_dir = exp_dir.joinpath(str1 + timestr + "_" + run._id)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    run["exp_dir"] = exp_dir

    '''LOG'''
    args = parse_args()
    run["args"] = args
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(f"Experiment directory: {exp_dir}")
    log_string('PARAMETER ...')
    log_string(args)

    run["dataset/train_and_test_artifacts"].track_files(args.data_path)

    train_file = f"{args.set_path}/train_set.txt"
    test_file = f"{args.set_path}/test_set.txt"
    assert os.path.exists(train_file)
    assert os.path.exists(test_file)
    run["dataset/train_file"].upload_files(train_file)
    run["dataset/test_file"].upload_files(test_file)

    #Data Augmentations
    augmentation_list = []
    if args.move_points is not None:
        max_movement_mm = args.move_points[0]
        prob_application = args.move_points[1]
        move_points = MovePointsXCoordinate(max_movement_mm=max_movement_mm, probability_application=prob_application, model=args.model)
        augmentation_list.append(move_points)
    if args.rotate_points is not None:
        angle_degree = args.rotate_points[0]
        prob_application = args.rotate_points[1]
        rotate_points = RotatePoints(max_degree=angle_degree, probability_application=prob_application, model=args.model)
        augmentation_list.append(rotate_points)
    if args.add_noise is not None:
        gaussian_noise = AddGaussianNoiseToYCoordinate(std=args.add_noise, model=args.model)
        augmentation_list.append(gaussian_noise)
    if args.remove_points is not None:
        percentage_points_removed = args.remove_points[0]
        prob_application = args.remove_points[1]
        remove_points = RemovePoints(percentage_removed_points=percentage_points_removed, probability_application=prob_application, model=args.model)
        augmentation_list.append(remove_points)
    if args.remove_cluster is not None:
        n_points_removed = args.remove_cluster[0]
        prob_application = args.remove_cluster[1]
        remove_cluster = RemoveCluster(n_points=n_points_removed, probability_application=prob_application, model=args.model)
        augmentation_list.append(remove_cluster)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.model == "lasernet_regr":
        train_dataset = WeldScansLasernetDataSet(transforms=augmentation_list,root=args.data_path, args=args, split='train', train_file=train_file, test_file=test_file)
        test_dataset = WeldScansLasernetDataSet(root=args.data_path, args=args, split='test', train_file=train_file, test_file=test_file)
    else:
        train_dataset = WeldScansPointnetDataSet(transforms=augmentation_list, root=args.data_path, args=args, split='train', train_file=train_file, test_file=test_file)
        test_dataset = WeldScansPointnetDataSet(root=args.data_path, args=args, split='test', train_file=train_file, test_file=test_file)

    #If we do cross-validation, the test set is copied from the train set
    if args.cv is not None:
        test_dataset = copy.deepcopy(train_dataset)

    #Save used python files
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_regression.py', str(exp_dir))
    shutil.copy('./Trainer.py', str(exp_dir))

    #Training
    trainer = Trainer(
        args=args, train_dataset=train_dataset, test_dataset=test_dataset,
        run=run, checkpoints_dir=checkpoints_dir, log_string=log_string, exp_dir=exp_dir
    )

    if args.cv is not None:
        trainer.run_training_pipeline_with_cv(num_folds=args.cv)
    else:
        trainer.run_training_pipeline()

    run.stop()

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {DEVICE} device.')
    args = parse_args()
    main(args)
