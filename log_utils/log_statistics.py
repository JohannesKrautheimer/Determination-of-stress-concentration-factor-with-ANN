import numpy as np
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

def log_instance_statistics_for_cv_fold(all_pred, all_targets, criterion, neptune_run, epoch, log_fn, split, fold):
    instance_loss = criterion(all_pred, all_targets)
    log_fn(f"{split.capitalize()} instance Loss for fold {fold} and epoch {epoch}: {instance_loss}")
    neptune_run[f"{split}/loss/fold_{fold}/instance_loss"].log(
        value=float(f"{instance_loss}"),
        step=epoch
    )
    
    mse_obj = MeanSquaredError()
    instance_mse = mse_obj(all_pred, all_targets)

    mae_obj = MeanAbsoluteError()
    instance_mae = mae_obj(all_pred, all_targets)

    r2_obj = R2Score()
    instance_r2 = r2_obj(all_pred, all_targets)

    log_fn(f"{split.capitalize()} instance MSE for fold {fold} and epoch {epoch}: {instance_mse}")
    log_fn(f"{split.capitalize()} instance MAE for fold {fold} and epoch {epoch}: {instance_mae}")
    log_fn(f"{split.capitalize()} instance R2 Score for fold {fold} and epoch {epoch}: {instance_r2}")
    neptune_run[f"{split}/mse/fold_{fold}/instance_mse"].log(
        value=float(f"{instance_mse}"),
        step=epoch
    )
    neptune_run[f"{split}/mae/fold_{fold}/instance_mae"].log(
        value=float(f"{instance_mae}"),
        step=epoch
    )
    neptune_run[f"{split}/r2_score/fold_{fold}/instance_r2_score"].log(
        value=float(f"{instance_r2}"),
        step=epoch
    )

    #Other statistics
    mean_pred_instance = np.mean(all_pred.numpy())
    variance_pred_instance = np.var(all_pred.numpy())
    mean_target_instance = np.mean(all_targets.numpy())
    variance_target_instance = np.var(all_targets.numpy())
    log_fn(f"{split.capitalize()} instance mean prediction for fold {fold} and epoch {epoch}: {mean_pred_instance}")
    log_fn(f"{split.capitalize()} instance variance prediction for fold {fold} and epoch {epoch}: {variance_pred_instance}")
    log_fn(f"{split.capitalize()} instance mean target for fold {fold} and epoch {epoch}: {mean_target_instance}")
    log_fn(f"{split.capitalize()} instance variance target for fold {fold} and epoch {epoch}: {variance_target_instance}")
    neptune_run[f"{split}/statistics/pred/fold_{fold}/instance_mean"].log(
        value=float(f"{mean_pred_instance}"),
        step=epoch
    )
    neptune_run[f"{split}/statistics/pred/fold_{fold}/instance_variance"].log(
        value=float(f"{variance_pred_instance}"),
        step=epoch
    )
    neptune_run[f"{split}/statistics/target/fold_{fold}/instance_mean"].log(
        value=float(f"{mean_target_instance}"),
        step=epoch
    )
    neptune_run[f"{split}/statistics/target/fold_{fold}/instance_variance"].log(
        value=float(f"{variance_target_instance}"),
        step=epoch
    )

    return instance_loss, instance_mse, instance_mae, instance_r2

def log_instance_statistics(all_pred, all_targets, criterion, neptune_run, epoch, log_fn, split):
    instance_loss = criterion(all_pred, all_targets)
    log_fn(f"{split.capitalize()} instance Loss for epoch {epoch}: {instance_loss}")
    neptune_run[f"{split}/loss/instance_loss"].log(
        value=float(f"{instance_loss}"),
        step=epoch
    )

    mse_obj = MeanSquaredError()
    instance_mse = mse_obj(all_pred, all_targets)

    mae_obj = MeanAbsoluteError()
    instance_mae = mae_obj(all_pred, all_targets)

    r2_obj = R2Score()
    instance_r2 = r2_obj(all_pred, all_targets)

    log_fn(f"{split.capitalize()} instance MSE for epoch {epoch}: {instance_mse}")
    log_fn(f"{split.capitalize()} instance MAE for epoch {epoch}: {instance_mae}")
    log_fn(f"{split.capitalize()} instance R2 Score for epoch {epoch}: {instance_r2}")
    neptune_run[f"{split}/mse/instance_mse"].log(
        value=float(f"{instance_mse}"),
        step=epoch
    )
    neptune_run[f"{split}/mae/instance_mae"].log(
        value=float(f"{instance_mae}"),
        step=epoch
    )
    neptune_run[f"{split}/r2_score/instance_r2_score"].log(
        value=float(f"{instance_r2}"),
        step=epoch
    )

    #Other statistics
    mean_pred_instance = np.mean(all_pred.numpy())
    variance_pred_instance = np.var(all_pred.numpy())
    mean_target_instance = np.mean(all_targets.numpy())
    variance_target_instance = np.var(all_targets.numpy())
    log_fn(f"{split.capitalize()} instance mean prediction for epoch {epoch}: {mean_pred_instance}")
    log_fn(f"{split.capitalize()} instance variance prediction for epoch {epoch}: {variance_pred_instance}")
    log_fn(f"{split.capitalize()} instance mean target for epoch {epoch}: {mean_target_instance}")
    log_fn(f"{split.capitalize()} instance variance target for epoch {epoch}: {variance_target_instance}")
    neptune_run[f"{split}/statistics/pred/instance_mean"].log(
        value=float(f"{mean_pred_instance}"),
        step=epoch
    )
    neptune_run[f"{split}/statistics/pred/instance_variance"].log(
        value=float(f"{variance_pred_instance}"),
        step=epoch
    )
    neptune_run[f"{split}/statistics/target/instance_mean"].log(
        value=float(f"{mean_target_instance}"),
        step=epoch
    )
    neptune_run[f"{split}/statistics/target/instance_variance"].log(
        value=float(f"{variance_target_instance}"),
        step=epoch
    )

    return instance_loss, instance_mse, instance_mae, instance_r2

#Log the batch statistics for a given fold of cross-validation
def log_batch_statistics_for_cv_fold(pred_cpu, target_cpu, run, batch_id, epoch, log_string, mse_per_batch, mae_per_batch, r2_per_batch, split, fold):
    #MSE
    mse_obj = MeanSquaredError()
    mse = mse_obj(pred_cpu, target_cpu)
    log_string(f"{split.capitalize()} MSE (Mean squared error) for fold {fold} batch id {batch_id} and epoch {epoch} is: {mse}")
    mse_per_batch.append(mse.clone().detach().cpu().numpy())
    run[f"{split}/mse/fold_{fold}/{epoch}/mse_per_batch"].log(
        value=float(f"{mse}"),
        step=batch_id
    )

    #MAE
    mae_obj = MeanAbsoluteError()
    mae = mae_obj(pred_cpu, target_cpu)
    log_string(f"{split.capitalize()} MAE (Mean absolute error) for fold {fold} batch id {batch_id} and epoch {epoch} is: {mae}")
    mae_per_batch.append(mae.clone().detach().cpu().numpy())
    run[f"{split}/mae/fold_{fold}/{epoch}/mae_per_batch"].log(
        value=float(f"{mae}"),
        step=batch_id
    )

    #R2 Score
    r2_obj = R2Score()
    r2_score = r2_obj(pred_cpu, target_cpu)
    log_string(f"{split.capitalize()} R2 Score for fold {fold} batch id {batch_id} and epoch {epoch} is: {r2_score}")
    r2_per_batch.append(r2_score.clone().detach().cpu().numpy())
    run[f"{split}/r2_score/fold_{fold}/{epoch}/r2_score_per_batch"].log(
        value=float(f"{r2_score}"),
        step=batch_id
    )

    #Other statistics
    mean_pred_batch = np.mean(pred_cpu.numpy())
    variance_pred_batch = np.var(pred_cpu.numpy())
    mean_target_batch = np.mean(target_cpu.numpy())
    variance_target_batch = np.var(target_cpu.numpy())
    run[f"{split}/statistics/pred/fold_{fold}/{epoch}/mean_per_batch"].log(
        value=float(f"{mean_pred_batch}"),
        step=batch_id
    )
    run[f"{split}/statistics/pred/fold_{fold}/{epoch}/variance_per_batch"].log(
        value=float(f"{variance_pred_batch}"),
        step=batch_id
    )
    run[f"{split}/statistics/target/fold_{fold}/{epoch}/mean_per_batch"].log(
        value=float(f"{mean_target_batch}"),
        step=batch_id
    )
    run[f"{split}/statistics/target/fold_{fold}/{epoch}/variance_per_batch"].log(
        value=float(f"{variance_target_batch}"),
        step=batch_id
    )

def log_batch_statistics(pred_cpu, target_cpu, run, batch_id, epoch, log_string, mse_per_batch, mae_per_batch, r2_per_batch, split):
    #MSE
    mse_obj = MeanSquaredError()
    mse = mse_obj(pred_cpu, target_cpu)
    log_string(f"{split.capitalize()} MSE (Mean squared error) for batch id {batch_id} and epoch {epoch} is: {mse}")
    mse_per_batch.append(mse.clone().detach().cpu().numpy())
    run[f"{split}/mse/{epoch}/mse_per_batch"].log(
        value=float(f"{mse}"),
        step=batch_id
    )

    #MAE
    mae_obj = MeanAbsoluteError()
    mae = mae_obj(pred_cpu, target_cpu)
    log_string(f"{split.capitalize()} MAE (Mean absolute error) for batch id {batch_id} and epoch {epoch} is: {mae}")
    mae_per_batch.append(mae.clone().detach().cpu().numpy())
    run[f"{split}/mae/{epoch}/mae_per_batch"].log(
        value=float(f"{mae}"),
        step=batch_id
    )

    #R2 Score
    r2_obj = R2Score()
    r2_score = r2_obj(pred_cpu, target_cpu)
    log_string(f"{split.capitalize()} R2 Score for batch id {batch_id} and epoch {epoch} is: {r2_score}")
    r2_per_batch.append(r2_score.clone().detach().cpu().numpy())
    run[f"{split}/r2_score/{epoch}/r2_score_per_batch"].log(
        value=float(f"{r2_score}"),
        step=batch_id
    )

    #Other statistics
    mean_pred_batch = np.mean(pred_cpu.numpy())
    variance_pred_batch = np.var(pred_cpu.numpy())
    mean_target_batch = np.mean(target_cpu.numpy())
    variance_target_batch = np.var(target_cpu.numpy())
    run[f"{split}/statistics/pred/{epoch}/mean_per_batch"].log(
        value=float(f"{mean_pred_batch}"),
        step=batch_id
    )
    run[f"{split}/statistics/pred/{epoch}/variance_per_batch"].log(
        value=float(f"{variance_pred_batch}"),
        step=batch_id
    )
    run[f"{split}/statistics/target/{epoch}/mean_per_batch"].log(
        value=float(f"{mean_target_batch}"),
        step=batch_id
    )
    run[f"{split}/statistics/target/{epoch}/variance_per_batch"].log(
        value=float(f"{variance_target_batch}"),
        step=batch_id
    )