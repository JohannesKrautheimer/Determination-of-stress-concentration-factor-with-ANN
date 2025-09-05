"""
Author: Öner Aydogan
Based on: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import argparse
import numpy as np
import json
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from sklearn.model_selection import KFold

from data_utils.WeldScansDataLoader import WeldScansLasernetDataSet, WeldScansPointnetDataSet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode (default: false)')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device (default: 0)')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size for testing (default: 24)')
    parser.add_argument('--num_point', type=int, default=None, help='Point Number (default: None)')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root (required)')
    parser.add_argument('--model', default='pointnet2_regr_msg', help='model name (default: pointnet2_regr_msg)')
    parser.add_argument('--process_data', action='store_true', default=False, help='process data before run instead of loading it on each iteration (default: false)')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals (default: false)')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling (default: false)')
    parser.add_argument('--data_path', type=str, default='data/weld_scans_2d_cuts_rotated_in_cm/', help='path to the location of the data (default: data/weld_scans_2d_cuts_rotated_in_cm/)')
    parser.add_argument('--set_path', type=str, default='train_test_sets/weld_scans_2d_cuts/default/', help='path to the directory with the test set config (default: train_test_sets/weld_scans_2d_cuts/default/)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cv', type=int, default=None, help='Apply k-fold cross validation with the given int as the number of folds')
    parser.add_argument('--dont_split_data', action='store_true', default=False, help='this can be activated for the case that the arg "cv" is used.\
        In this case the cv is done but the dataset is not split for each fold, instead the whole data is used. This is useful e.g. if you want to evaluate the different folds on one full test dataset')
    return parser.parse_args()

def print_batch_statistics(pred, target, log_string, batch_id, fold=None):
    fold_string = "" if fold is None else f" and fold {fold}"

    #MSE
    mse_obj = MeanSquaredError()
    mse = mse_obj(pred, target)
    log_string(f"Test MSE (Mean squared error) for batch id {batch_id}{fold_string} is: {mse}")

    #MAE
    mae_obj = MeanAbsoluteError()
    mae = mae_obj(pred, target)
    log_string(f"Test MAE (Mean absolute error) for batch id {batch_id}{fold_string} is: {mae}")

    #R2 Score
    r2_obj = R2Score()
    r2_score = r2_obj(pred, target)
    log_string(f"Test R2 Score for batch id {batch_id}{fold_string} is: {r2_score}")

    #Other statistics
    mean_pred_batch = np.mean(pred.numpy())
    variance_pred_batch = np.var(pred.numpy())
    mean_target_batch = np.mean(target.numpy())
    variance_target_batch = np.var(target.numpy())
    log_string(f"Mean prediction for batch id {batch_id}{fold_string}: {mean_pred_batch}")
    log_string(f"Variance prediction for batch id {batch_id}{fold_string}: {variance_pred_batch}")
    log_string(f"Mean target for batch id {batch_id}{fold_string}: {mean_target_batch}")
    log_string(f"Variance target for batch id {batch_id}{fold_string}: {variance_target_batch}")

def print_instance_statistics(all_pred, all_targets, log_string, fold=None):
    fold_string = "" if fold is None else f" for fold {fold}"

    mse_obj = MeanSquaredError()
    instance_mse = mse_obj(all_pred, all_targets)

    mae_obj = MeanAbsoluteError()
    instance_mae = mae_obj(all_pred, all_targets)

    r2_obj = R2Score()
    instance_r2 = r2_obj(all_pred, all_targets)

    #Other statistics
    mean_pred_instance = np.mean(all_pred.numpy())
    variance_pred_instance = np.var(all_pred.numpy())
    mean_target_instance = np.mean(all_targets.numpy())
    variance_target_instance = np.var(all_targets.numpy())
    log_string(f"Mean prediction instance{fold_string}: {mean_pred_instance}")
    log_string(f"Variance prediction instance{fold_string}: {variance_pred_instance}")
    log_string(f"Mean target instance{fold_string}: {mean_target_instance}")
    log_string(f"Variance target instance{fold_string}: {variance_target_instance}")

    return instance_mse, instance_mae, instance_r2

def init_model(args, log_string, experiment_dir, fold=None):
    log_string('Load model...')
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        
    if fold is not None:
        checkpoint = torch.load(str(experiment_dir) + f'/checkpoints/best_model_fold_{fold}.pth', map_location=device)
    else:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=device)

    max_len = None
    if args.model == "lasernet_regr":
        max_len = checkpoint['max_len']
        regressor = model.get_model(max_len)
    else:
        if "num_point" in checkpoint:
            args.num_point = checkpoint["num_point"]
        assert args.num_point is not None
        log_string(f"Num_point: {args.num_point}")
        regressor = model.get_model(normal_channel=args.use_normals)

    if not args.use_cpu:
        regressor = regressor.cuda()

    regressor.load_state_dict(checkpoint['model_state_dict'])

    seed = None
    if "seed" in checkpoint:
        seed = checkpoint["seed"]
    if args.seed is not None:
        seed = args.seed
    if seed is not None:
        log_string(f"Seed: {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)

    return regressor, max_len

def get_data_loader(dataset):
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    return dataLoader 

# Returns: instance_mse, instance_mae, instance_r2, result_dict: filename -> {pred, target}
def test(model, loader, log_string, args, fold=None):
    with torch.no_grad():
        all_pred = None
        all_targets = None
        result_dict = {} # filename -> (pred, target)
        regressor = model.eval()

        for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader)):
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            if args.model != "lasernet_regr":
                points = points.transpose(2, 1)
            pred = regressor(points)
            pred = torch.flatten(pred).clone().detach().cpu()
            target = torch.flatten(target).clone().detach().cpu()

            if all_pred is None:
                    all_pred = pred
            else:
                all_pred = torch.cat((all_pred, pred), 0)
            if all_targets is None:
                all_targets = target
            else:
                all_targets = torch.cat((all_targets, target), 0)

            print_batch_statistics(pred, target, log_string, batch_id, fold)

            #Assign the filenames to the corresponding prediction & target
            if fold is None:
                datapaths = loader.dataset.datapath
            else:
                datapaths = [loader.dataset.dataset.datapath[idx] for idx in loader.dataset.indices]
            filenames = [entry[1] for entry in datapaths[batch_id * args.batch_size:(batch_id*args.batch_size + args.batch_size)]]
            for idx, filename in enumerate(filenames):
                result_dict[filename] = {"pred": float(pred[idx]), "target": float(target[idx])}

        instance_mse, instance_mae, instance_r2 = print_instance_statistics(all_pred, all_targets, log_string, fold)

        log_string(f"All predictions are: {all_pred}")
        log_string(f"All targets are: {all_targets}")

    return instance_mse, instance_mae, instance_r2, result_dict
        
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/regression/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''MODEL LOADING'''
    fold = None
    if args.cv:
        fold = 0
    regressor, max_len = init_model(args, log_string, experiment_dir, fold)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    if args.cv:
        train_file = f"{args.set_path}/train_set.txt" 
    test_file = f"{args.set_path}/test_set.txt"

    if args.model == "lasernet_regr":
        test_dataset = WeldScansLasernetDataSet(root=args.data_path, args=args, split='test', test_file=test_file, max_sequence_len=max_len)
        if args.cv:
            train_dataset = WeldScansLasernetDataSet(root=args.data_path, args=args, split='train', train_file=train_file, test_file=test_file, max_sequence_len=max_len)
    else:
        test_dataset = WeldScansPointnetDataSet(root=args.data_path, args=args, split='test', test_file=test_file)
        if args.cv:
            train_dataset = WeldScansPointnetDataSet(root=args.data_path, args=args, split='train', train_file=train_file, test_file=test_file)


    #With cross-validation. Hereby only the train set is considered and for each fold the corresponding fold test set is used.
    if args.cv:
        cross_validations_results_dict = {}
        test_result_dict = {}
        kf = KFold(n_splits=args.cv, shuffle=True)
        for fold, (train_index, test_index) in tqdm(enumerate(kf.split(train_dataset)), total=args.cv):
            log_string(f"Fold: {fold}")
            #Use all data from train dataset as test set
            if args.dont_split_data:
                train_index = np.array([None] * len(train_dataset))
                test_index = np.array(list(range(len(train_dataset))))
            # Reset model for each fold after the first one
            if fold > 0:
                regressor, max_len = init_model(args, log_string, experiment_dir, fold)
            fold_test_dataset = torch.utils.data.Subset(train_dataset, test_index)
            fold_testDataLoader = get_data_loader(fold_test_dataset)

            # test_result_dict: fold: filename -> {pred, target}
            instance_mse, instance_mae, instance_r2, fold_test_result_dict = test(regressor.eval(), fold_testDataLoader, log_string, args, fold)        
            log_string(f"Test Instance MSE for fold {fold}: {instance_mse}")
            log_string(f"Test Instance MAE for fold {fold}: {instance_mae}")
            log_string(f"Test Instance R2 for fold {fold}: {instance_r2}")
            test_result_dict[f"fold_{fold}"] = fold_test_result_dict

            fold_test_results = {
                            "instance_mse": instance_mse,
                            "instance_mae": instance_mae,
                            "instance_r2": instance_r2,
                        }
            cross_validations_results_dict[f"fold_{fold}"] = {
                "test_results": fold_test_results,
            }

        # Calc stats of CV
        all_folds_instance_mse = [cross_validations_results_dict[fold]['test_results']['instance_mse'] for fold in cross_validations_results_dict.keys()]
        all_folds_instance_mae = [cross_validations_results_dict[fold]['test_results']['instance_mae'] for fold in cross_validations_results_dict.keys()]
        all_folds_instance_r2 = [cross_validations_results_dict[fold]['test_results']['instance_r2'] for fold in cross_validations_results_dict.keys()]

        mean_test_mse, std_test_mse = np.mean(all_folds_instance_mse), np.std(all_folds_instance_mse)
        mean_test_mae, std_test_mae = np.mean(all_folds_instance_mae), np.std(all_folds_instance_mae)
        mean_test_r2, std_test_r2 = np.mean(all_folds_instance_r2), np.std(all_folds_instance_r2)


        cross_validations_results_dict["all_folds"] = {
            "mean_test_mse": mean_test_mse,
            "mean_test_mae": mean_test_mae,
            "mean_test_r2": mean_test_r2,
            "std_test_mse": std_test_mse,
            "std_test_mae": std_test_mae,
            "std_test_r2": std_test_r2,
        }

        log_string("Statistics over all folds:")
        log_string(f"Mean test MSE: {mean_test_mse}      Standard deviation test MSE: {std_test_mse}")
        log_string(f"Mean test MAE: {mean_test_mae}      Standard deviation test MAE: {std_test_mae}")
        log_string(f"Mean test R2: {mean_test_r2}        Standard deviation test R2: {std_test_r2}")
    else:
        testDataLoader = get_data_loader(test_dataset)
        # test_result_dict: filename -> {pred, target}
        instance_mse, instance_mae, instance_r2, test_result_dict = test(regressor.eval(), testDataLoader, log_string, args)
        log_string(f"Test Instance MSE: {instance_mse}")
        log_string(f"Test Instance MAE: {instance_mae}")
        log_string(f"Test Instance R2: {instance_r2}")

    #Save test results
    with open(f"test_results/{args.log_dir.split('/')[-1]}_test_result_dict.json", 'w') as f:
        json.dump(test_result_dict, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
