"""
Author: Öner Aydogan
Based on: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import torch
from tqdm import tqdm
from log_utils.log_statistics import log_batch_statistics, log_batch_statistics_for_cv_fold, log_instance_statistics, log_instance_statistics_for_cv_fold

def test_model(model, data_loader, criterion, run, epoch, log_string, args, fold=None):
    all_pred = None
    all_targets = None
    result_dict = {} # filename -> (pred, target)
    regressor = model.eval()
    criterion.eval()

    for batch_id, (points, target) in tqdm(enumerate(data_loader), total=len(data_loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        if args.model != "lasernet_regr":
            points = points.transpose(2, 1)
        pred = regressor(points)
        pred_cpu = torch.flatten(pred).clone().detach().cpu()
        target_cpu = torch.flatten(target).clone().detach().cpu()

        if all_pred is None:
                all_pred = pred_cpu
        else:
            all_pred = torch.cat((all_pred, pred_cpu), 0)
        if all_targets is None:
            all_targets = target_cpu
        else:
            all_targets = torch.cat((all_targets, target_cpu), 0)

        loss = criterion(pred, target)
        if fold is not None:
            log_string(f"Test Loss for fold {fold} batch id {batch_id} and epoch {epoch} is: {float(loss)}")
            run[f"test/loss/fold_{fold}/{epoch}/loss_per_batch"].log(
                value=float(f"{loss}"),
                step=batch_id
            )
        else:    
            log_string(f"Test Loss for batch id {batch_id} and epoch {epoch} is: {float(loss)}")
            run[f"test/loss/{epoch}/loss_per_batch"].log(
                value=float(f"{loss}"),
                step=batch_id
            )

        if fold is not None:
            log_batch_statistics_for_cv_fold(pred_cpu, target_cpu, run, batch_id, epoch, log_string, mse_per_batch=[], mae_per_batch=[], r2_per_batch=[], split="test", fold=fold)
        else:
            log_batch_statistics(pred_cpu, target_cpu, run, batch_id, epoch, log_string, mse_per_batch=[], mae_per_batch=[], r2_per_batch=[], split="test")

        #Assign the filenames to the corresponding prediction & target
        if fold is None:
            datapaths = data_loader.dataset.datapath
        else:
            datapaths = [data_loader.dataset.dataset.datapath[idx] for idx in data_loader.dataset.indices]
        filenames = [entry[1] for entry in datapaths[batch_id * args.batch_size:(batch_id*args.batch_size + args.batch_size)]]
        for idx, filename in enumerate(filenames):
            result_dict[filename] = {"pred": float(pred[idx]), "target": float(target[idx])}

    if fold is not None:
        instance_loss, instance_mse, instance_mae, instance_r2 = log_instance_statistics_for_cv_fold(all_pred, all_targets, criterion, run, epoch, log_fn=log_string, split="test", fold=fold)
    else:
        instance_loss, instance_mse, instance_mae, instance_r2 = log_instance_statistics(all_pred, all_targets, criterion, run, epoch, log_fn=log_string, split="test")

    return instance_loss, instance_mse, instance_mae, instance_r2, result_dict