import numpy as np
import pandas as pd
from properscoring import crps_ensemble
import torch
import ot


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse  = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr




def compute_crps_from_samples(forecast_samples, ground_truth, axis=-1):
    """
    ----------
    forecast_samples : np.ndarray
        Forecast samples of shape [num_samples, num_observations, horizon].
    ground_truth : np.ndarray
        True observations of shape [num_observations, horizon].
    axis : int, optional
        Axis corresponding to the samples in the final reshaped array.
        Default is -1 (samples in the last dimension).
    """
    # Move samples to last axis: [num_obs, horizon, num_samples]
    forecasts = np.transpose(forecast_samples, (1, 2, 0))

    # Check dimensions
    assert forecasts.shape[:2] == ground_truth.shape, \
        f"Shape mismatch: forecasts {forecasts.shape[:2]}, ground_truth {ground_truth.shape}"


    crps = crps_ensemble(ground_truth, forecasts, axis=axis)

    return np.mean(crps), np.std(crps)


def get_mse_mae(preds, trues):
  mean_preds = np.mean(preds, axis=0)
  mae, mse, _, _, _, _, _ = metric(mean_preds, trues)
  return mse, mae


def emd_loss(x, y):

    a = torch.tensor(ot.unif(x.shape[0]), dtype=torch.float32).to(x.device)
    b = torch.tensor(ot.unif(y.shape[0]), dtype=torch.float32).to(y.device)

    M = ot.dist(x, y, metric='euclidean') ** 2

    T = ot.emd(a, b, M)
    cost = torch.sum(T * M)

    return cost


def safe_crps(predictions, groundtruths, batch_size=10):
    n = predictions.shape[0]
    crps_vals = []
    for i in range(0, n, batch_size):
        batch_preds = predictions[i:i+batch_size]
        crps, _ = compute_crps_from_samples(batch_preds, groundtruths)
        crps_vals.append(crps)
    return np.mean(crps_vals), np.std(crps_vals)

def fc_metrics(predictions, groundtruths, setting, batch_size=10):
    predictions = predictions.astype(np.float32)
    groundtruths = groundtruths.astype(np.float32)
    
    mean_preds = np.mean(predictions, axis=0)
    mae, mse, _, _, _, _, _ = metric(mean_preds, groundtruths)
    crps, crps_std = safe_crps(predictions, groundtruths)
    print("=" * 40)
    print("Final results")
    print(f"MSE: {mse:.5f}, MAE: {mae:.5f}, CRPS: {crps:.5f}, CRPS_STD: {crps_std:.5f}")
    f = open("result.txt", 'a')
    f.write(setting + "  \n")
    f.write(f"MSE: {mse:.5f}, MAE: {mae:.5f}, CRPS: {crps:.5f}, CRPS_STD: {crps_std:.5f}")
    f.write('\n')
    f.write('\n')
    f.close()
    print("=" * 40)


    