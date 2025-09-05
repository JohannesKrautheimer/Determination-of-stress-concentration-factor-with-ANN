import torch
import copy
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from Tester import test_model
from log_utils.log_statistics import log_batch_statistics, log_batch_statistics_for_cv_fold, log_instance_statistics, log_instance_statistics_for_cv_fold
import provider
import importlib
import os
import json
import matplotlib.pyplot as plt

# def plot_pc_tensor(tensor, label=""):
#     coordinates = tensor[0, :, :2].detach().numpy()
#     plt.scatter(coordinates[:, 0], coordinates[:, 1], label=label)
#     plt.legend()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

class Trainer():
    def __init__(self, args, train_dataset, test_dataset, run, checkpoints_dir, log_string, exp_dir):
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.global_epoch = 0
        self.run = run
        self.checkpoints_dir = checkpoints_dir
        self.log_string = log_string
        self.exp_dir = exp_dir
        self.init_regressor_optimizer_and_scheduler()

    def get_data_loader(self, train_dataset, test_dataset):
        trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=1, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=1)
        return trainDataLoader, testDataLoader

    def init_optimizer(self):
        if self.args.optimizer == 'Adam':
            optimizer_args = {
                "lr": self.args.learning_rate,
                "betas": (self.args.b1, 0.999),
                "eps": 1e-08,
                "weight_decay": self.args.decay_rate
            }
            self.optimizer = torch.optim.Adam(
                self.regressor.parameters(),
                **optimizer_args
            )
        else:
            optimizer_args = {
                "lr": 0.01,
                "momentum": 0.9
            }
            self.optimizer = torch.optim.SGD(self.regressor.parameters(), **optimizer_args)
        self.run["train/optimizer_args"] = optimizer_args

    def init_scheduler(self):
        scheduler_args = {
            "step_size": 20,
            "gamma": 0.7
        }
        self.run["train/scheduler_args"] = scheduler_args
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **scheduler_args)

    def init_regressor(self):
        model = importlib.import_module(self.args.model)
        self.log_string(model)

        max_len = -1
        if self.args.model == "lasernet_regr":
            # max_len = self.regressor.input_seq_length
            max_len = self.train_dataset.max_len if self.train_dataset.max_len > self.test_dataset.max_len else self.test_dataset.max_len
            self.log_string(f"Max length for Lasernet model: {max_len}")
            new_regressor = model.get_model(max_len)    
        else:
            new_regressor = model.get_model(normal_channel=self.args.use_normals)
        self.regressor = new_regressor
        self.criterion = model.get_loss(self.args.loss_fn)
        #TODO: is this neccessary?
        self.regressor.apply(inplace_relu)

        if not self.args.use_cpu:
            self.regressor = self.regressor.cuda()
            self.criterion = self.criterion.cuda()
        
        if os.path.isfile(str(self.exp_dir) + '/checkpoints/best_model.pth'):
            checkpoint = torch.load(str(self.exp_dir) + '/checkpoints/best_model.pth')
            self.start_epoch = checkpoint['epoch']
            self.regressor.load_state_dict(checkpoint['model_state_dict'])
            self.log_string('Use pretrained model')
        else:
            self.log_string('No existing model, starting training from scratch...')
            self.start_epoch = 0
        
        self.run["train/start_epoch"] = self.start_epoch

    def init_regressor_optimizer_and_scheduler(self):
        self.init_regressor()
        self.init_optimizer()
        self.init_scheduler()

    def prepare_points_for_pointnet(self, points):
        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        # I commented this out because we do our own random shifting as augmentations 
        # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        return points

    def run_training_pipeline(self):
        trainDataLoader, testDataLoader = self.get_data_loader(self.train_dataset, self.test_dataset)

        best_test_instance_loss = 1000000

        self.log_string('Start training...')
        for epoch in range(self.start_epoch, self.args.epoch):
            self.log_string(f"Epoch {self.global_epoch + 1} ({epoch + 1}/{self.args.epoch}):")
            all_pred = None
            all_targets = None
            mse_per_batch = []
            mae_per_batch = []
            r2_per_batch = []
            self.regressor.train()
            self.criterion.train()

            self.scheduler.step()
            for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                self.optimizer.zero_grad()

                if self.args.model != "lasernet_regr":
                    points = self.prepare_points_for_pointnet(points)

                if not self.args.use_cpu:
                    points, target = points.cuda(), target.cuda()

                pred= self.regressor(points)
                pred_cpu = torch.flatten(pred.clone()).detach().cpu()
                target_cpu = torch.flatten(target.clone()).detach().cpu()
                if all_pred is None:
                    all_pred = pred_cpu
                else:
                    all_pred = torch.cat((all_pred, pred_cpu), 0)
                if all_targets is None:
                    all_targets = target_cpu
                else:
                    all_targets = torch.cat((all_targets, target_cpu), 0)
                #Öner: I changed long to float
                # loss = criterion(pred, target.long(), trans_feat)
                loss = self.criterion(pred, target)
                # loss.requires_grad = True

                self.log_string(f"Train Loss for batch id {batch_id} and epoch {epoch} is: {float(loss)}")
                self.run[f"train/loss/{epoch}/loss_per_batch"].log(
                    value=float(f"{loss}"),
                    step=batch_id
                )

                log_batch_statistics(pred_cpu, target_cpu, self.run, batch_id, epoch, self.log_string, mse_per_batch, mae_per_batch, r2_per_batch, split="train")

                loss.backward()
                self.optimizer.step()

            log_instance_statistics(all_pred, all_targets, self.criterion, self.run, epoch, log_fn=self.log_string, split="train")

            #Evaluate model
            with torch.no_grad():
                test_instance_loss, test_instance_mse, test_instance_mae, test_instance_r2, test_result_dict = test_model(self.regressor, testDataLoader, self.criterion, self.run, epoch, self.log_string, self.args)
                is_new_best_instance_error = test_instance_loss < best_test_instance_loss

                if (is_new_best_instance_error):
                    best_test_instance_loss = test_instance_loss
                    best_epoch = epoch + 1

                    self.log_string('Save model...')
                    savepath = str(self.checkpoints_dir) + '/best_model.pth'
                    self.log_string(f"New best epoch: {best_epoch} with loss {best_test_instance_loss}")
                    self.log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'instance_loss': best_test_instance_loss,
                        'loss_function': self.args.loss_fn,
                        'model_state_dict': self.regressor.state_dict(),
                        'seed': self.args.seed
                    }
                    state_neptune = {
                        'epoch': best_epoch,
                        'instance_loss': best_test_instance_loss,
                        'loss_function': self.args.loss_fn,
                        'seed': self.args.seed
                    }
                    if self.args.model == "lasernet_regr":
                        state['max_len'] = self.regressor.input_seq_length
                    if self.args.model == "pointnet2_regr_msg":
                        state['num_point'] = self.args.num_point
                    torch.save(state, savepath)
                    self.run["best_model/state"] = state_neptune

                    with open(f"{self.exp_dir}/test_result_dict.json", 'w') as f:
                            json.dump(test_result_dict, f, indent=4)

                self.log_string(f"Best Test Instance Loss ({self.args.loss_fn}) so far: {best_test_instance_loss}")

                self.global_epoch += 1
        self.log_string('End of training...')

    # Train the model using k-fold cross-validation
    def run_training_pipeline_with_cv(self, num_folds):
        '''TRANING'''
        self.log_string('Start training...')
        
        # cross_validations_results_dict: 
        #   {
        #     'fold_<fold_number>': {
        #                 train_results: {
        #                     "instance_loss", "instance_mse", "instance_mae", "instance_r2", "epoch"
        #                 },
        #                 test_results: <same as train_results> ...
        #             },
        #     'all_folds': {
        #         mean_test_loss, std_test_loss, ... (same for mse, mae and r2)
        #     }
        #   }
        cross_validations_results_dict = {}

        kf = KFold(n_splits=num_folds, shuffle=True)
        for fold, (train_index, test_index) in tqdm(enumerate(kf.split(self.train_dataset)), total=self.args.cv):
            # Reset model for each fold after the first one
            if fold != 0:
                self.init_regressor_optimizer_and_scheduler()

            best_test_instance_loss = 1000000

            self.log_string(f"Fold: {fold}")

            fold_train_dataset = torch.utils.data.Subset(self.train_dataset, train_index)
            fold_test_dataset = torch.utils.data.Subset(self.test_dataset, test_index)
            fold_test_dataset.dataset.transforms = []
            #We don't do augmentations on the test set
            trainDataLoader, testDataLoader = self.get_data_loader(fold_train_dataset, fold_test_dataset)

            for epoch in range(self.start_epoch, self.args.epoch):
                self.log_string(f"Epoch ({epoch + 1}/{self.args.epoch}):")
                all_pred = None
                all_targets = None
                mse_per_batch = []
                mae_per_batch = []
                r2_per_batch = []
                self.regressor.train()
                self.criterion.train()

                self.scheduler.step()
                for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                    self.optimizer.zero_grad()

                    # plot_pc_tensor(points, label="before augments")
                    # plt.figure()
                    if self.args.model != "lasernet_regr":
                        points = self.prepare_points_for_pointnet(points)

                    if not self.args.use_cpu:
                        points, target = points.cuda(), target.cuda()

                    pred= self.regressor(points)
                    pred_cpu = torch.flatten(pred.clone()).detach().cpu()
                    target_cpu = torch.flatten(target.clone()).detach().cpu()
                    if all_pred is None:
                        all_pred = pred_cpu
                    else:
                        all_pred = torch.cat((all_pred, pred_cpu), 0)
                    if all_targets is None:
                        all_targets = target_cpu
                    else:
                        all_targets = torch.cat((all_targets, target_cpu), 0)
                    #Öner: I changed long to float
                    # loss = criterion(pred, target.long(), trans_feat)
                    loss = self.criterion(pred, target)
                    # loss.requires_grad = True

                    self.log_string(f"Train Loss for fold {fold} batch id {batch_id} and epoch {epoch} is: {float(loss)}")
                    self.run[f"train/loss/fold_{fold}/{epoch}/loss_per_batch"].log(
                        value=float(f"{loss}"),
                        step=batch_id
                    )

                    log_batch_statistics_for_cv_fold(pred_cpu, target_cpu, self.run, batch_id, epoch, self.log_string, mse_per_batch, mae_per_batch, r2_per_batch, split="train", fold=fold)

                    loss.backward()
                    self.optimizer.step()

                train_instance_loss, train_instance_mse, train_instance_mae, train_instance_r2 = log_instance_statistics_for_cv_fold(all_pred, all_targets, self.criterion, self.run, epoch, log_fn=self.log_string, split="train", fold=fold)

                #Evaluate model
                with torch.no_grad():
                    test_instance_loss, test_instance_mse, test_instance_mae, test_instance_r2, test_result_dict = test_model(self.regressor, testDataLoader, self.criterion, self.run, epoch, self.log_string, self.args, fold=fold)
                    is_new_best_instance_error = test_instance_loss < best_test_instance_loss

                    if (is_new_best_instance_error):
                        fold_train_results = {
                            "instance_loss": train_instance_loss,
                            "instance_mse": train_instance_mse,
                            "instance_mae": train_instance_mae,
                            "instance_r2": train_instance_r2,
                            "epoch": epoch
                        }
                        fold_test_results = {
                            "instance_loss": test_instance_loss,
                            "instance_mse": test_instance_mse,
                            "instance_mae": test_instance_mae,
                            "instance_r2": test_instance_r2,
                            "epoch": epoch
                        }
                        cross_validations_results_dict[f"fold_{fold}"] = {
                            "train_results": fold_train_results,
                            "test_results": fold_test_results,
                        }
                        best_test_instance_loss = test_instance_loss
                        best_epoch = epoch + 1

                        savepath = str(self.checkpoints_dir) + f'/best_model_fold_{fold}.pth'
                        self.log_string(f"Fold {fold} new best epoch: {best_epoch} with loss {best_test_instance_loss}")
                        self.log_string('Save model...')
                        self.log_string('Saving at %s' % savepath)
                        state = {
                            'epoch': best_epoch,
                            'instance_loss': best_test_instance_loss,
                            'loss_function': self.args.loss_fn,
                            'model_state_dict': self.regressor.state_dict(),
                            'seed': self.args.seed
                        }
                        state_neptune = {
                        'epoch': best_epoch,
                        'instance_loss': best_test_instance_loss,
                        'loss_function': self.args.loss_fn,
                        'seed': self.args.seed
                        }
                        if self.args.model == "lasernet_regr":
                            state['max_len'] = self.regressor.input_seq_length
                        if self.args.model == "pointnet2_regr_msg":
                            state['num_point'] = self.args.num_point
                        torch.save(state, savepath)
                        self.run[f"best_model/fold_{fold}/state"] = state_neptune

                        with open(f"{self.exp_dir}/fold_{fold}_test_result_dict.json", 'w') as f:
                            json.dump(test_result_dict, f, indent=4)
                            

                self.log_string(f"Best Test Instance Loss ({self.args.loss_fn}) for fold {fold} so far: {best_test_instance_loss}")

                self.global_epoch += 1
    
        self.log_string('End of training...')

        # Calc stats of CV
        all_folds_instance_loss = [cross_validations_results_dict[fold]['test_results']['instance_loss'] for fold in cross_validations_results_dict.keys()]
        all_folds_instance_mse = [cross_validations_results_dict[fold]['test_results']['instance_mse'] for fold in cross_validations_results_dict.keys()]
        all_folds_instance_mae = [cross_validations_results_dict[fold]['test_results']['instance_mae'] for fold in cross_validations_results_dict.keys()]
        all_folds_instance_r2 = [cross_validations_results_dict[fold]['test_results']['instance_r2'] for fold in cross_validations_results_dict.keys()]

        mean_test_loss, std_test_loss = np.mean(all_folds_instance_loss), np.std(all_folds_instance_loss)
        mean_test_mse, std_test_mse = np.mean(all_folds_instance_mse), np.std(all_folds_instance_mse)
        mean_test_mae, std_test_mae = np.mean(all_folds_instance_mae), np.std(all_folds_instance_mae)
        mean_test_r2, std_test_r2 = np.mean(all_folds_instance_r2), np.std(all_folds_instance_r2)


        cross_validations_results_dict["all_folds"] = {
            "mean_test_loss": mean_test_loss,
            "mean_test_mse": mean_test_mse,
            "mean_test_mae": mean_test_mae,
            "mean_test_r2": mean_test_r2,
            "std_test_loss": std_test_loss,
            "std_test_mse": std_test_mse,
            "std_test_mae": std_test_mae,
            "std_test_r2": std_test_r2,
        }

        self.run["cross_validation_results"] = cross_validations_results_dict

        self.log_string("Statistics over all folds:")
        self.log_string(f"Mean test Loss: {mean_test_loss}    Standard deviation test Loss: {std_test_loss}")
        self.log_string(f"Mean test MSE: {mean_test_mse}      Standard deviation test MSE: {std_test_mse}")
        self.log_string(f"Mean test MAE: {mean_test_mae}      Standard deviation test MAE: {std_test_mae}")
        self.log_string(f"Mean test R2: {mean_test_r2}        Standard deviation test R2: {std_test_r2}")