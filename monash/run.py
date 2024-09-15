import math
import sys
sys.path.append("../")

import einops
import numpy as np
import pandas as pd
import torch

import gluonts_util
from visionts import VisionTS

POSSIBLE_SEASONALITIES = {
    "S": [3600],  # 1 hour
    "T": [1440, 10080],  # 1 day or 1 week
    "H": [24],  # 1 day or 1 week
    "D": [7, 30, 365],  # 1 week, 1 month or 1 year
    "W": [52, 4], # 1 year or 1 month
    "M": [12, 6, 3], # 3 months, 6 months or 1 year
    "B": [5],
    "Q": [4, 2], # 6 months or 1 year
}


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]
    return base_freq


def get_seasonality_list(freq: str) -> int:
    offset = pd.tseries.frequencies.to_offset(freq)
    base_seasonality_list = POSSIBLE_SEASONALITIES.get(norm_freq_str(offset.name), 1)
    seasonality_list = []
    for base_seasonality in base_seasonality_list:
        seasonality, remainder = divmod(base_seasonality, offset.n)
        if not remainder:
            seasonality_list.append(seasonality)
    seasonality_list.append(1)
    return seasonality_list


def compute_mae_with_nan(y_true, y_pred):
    mae = []
    for i in range(y_true.size(0)):
        diff = torch.abs(y_pred[i, :, 0] - y_true[i, :, 0])
        if torch.isnan(diff).all():
            continue
        diff = diff[~torch.isnan(diff)].mean().item()
        mae.append(diff)
    return np.asarray(mae)


def evaluate_mae(model: VisionTS, train_list: list, test_list: list, batch_size, device, periodicity):
    # We combine testing data with the context lengths
    seq_len_to_group_data = {}
    for i in range(len(train_list)):
        train_len = len(train_list[i])
        if train_len not in seq_len_to_group_data:
            seq_len_to_group_data[train_len] = [[], []]
        seq_len_to_group_data[train_len][0].append(train_list[i])
        seq_len_to_group_data[train_len][1].append(test_list[i])
    
    mae_list = []
    for train_len in seq_len_to_group_data:
        cur_train, cur_test = seq_len_to_group_data[train_len]
        convert = lambda array: torch.FloatTensor(
            einops.rearrange(np.array(array), 'b t -> b t 1')
        ).to(device)
        cur_train = convert(cur_train)
        cur_test = convert(cur_test)
        context_len = cur_train.shape[1]
        pred_len = cur_test.shape[1]
        model.update_config(context_len=context_len, pred_len=pred_len, periodicity=periodicity)

        for batch_i in range(int(math.ceil(len(cur_train) / batch_size))):
            batch_start = batch_i * batch_size
            if batch_start >= len(cur_train):
                continue
            batch_end = batch_start + batch_size
            if batch_end > len(cur_train):
                batch_end = len(cur_train)

            cur_batch_train = cur_train[batch_start:batch_end]
            cur_batch_test = cur_test[batch_start:batch_end]
            with torch.no_grad():
                cur_batch_pred = model(cur_batch_train, fp64=True) # [b t 1]
            cur_mae = compute_mae_with_nan(cur_batch_pred, cur_batch_test)
            mae_list.append(cur_mae)
    return np.mean(np.concatenate(mae_list))


def convert_context_len(context_len, no_periodicity_context_len, periodicity):
    if periodicity == 1:
        context_len = no_periodicity_context_len
    # Round context length to the integer multiples of the period
    context_len = int(round(context_len / periodicity)) * periodicity
    return context_len


if __name__ == "__main__":

    device = 'cuda:2'
    # device = 'cpu'
    batch_size = 1024
    dataset = ['m1_monthly', 'monash_m3_monthly', 'monash_m3_other', 'm4_monthly', 'm4_weekly', 'm4_daily', 'm4_hourly', 'tourism_quarterly', 'tourism_monthly', 'cif_2016_6', 'cif_2016_12', 'australian_electricity_demand', 'bitcoin', 'pedestrian_counts', 'vehicle_trips_without_missing', 'kdd_cup_2018_without_missing', 'weather', 'nn5_daily_without_missing', 'nn5_weekly', 'car_parts_without_missing', 'fred_md', 'traffic_hourly', 'traffic_weekly', 'rideshare_without_missing', 'hospital', 'covid_deaths', 'temperature_rain_without_missing', 'sunspot_without_missing', 'saugeenday', 'us_births']

    model = VisionTS(ckpt_dir="../ckpt/").to(device)

    for dataset_i, dataset in enumerate(dataset):
        test_data, metadata = gluonts_util.get_gluonts_test_dataset(dataset)
        data_train = [x['target'] for x in test_data.input]
        data_test = [x['target'] for x in test_data.label]
        pred_len = len(data_test[0])

        seasonality_list = get_seasonality_list(metadata.freq)
        best_valid_mae = float('inf')
        best_valid_p = 1
        for periodicity in seasonality_list:
            context_len = convert_context_len(1000, 300, periodicity)

            val_train = [x[-context_len-pred_len:-pred_len] for x in data_train]
            val_test = [x[-pred_len:] for x in data_train]
            val_mae = evaluate_mae(model, val_train, val_test, batch_size, device, periodicity)
            if val_mae < best_valid_mae:
                best_valid_p = periodicity
                best_valid_mae = val_mae
                print(f"autotune: P = {periodicity} | valid mae = {val_mae}, accept!")
            else:
                print(f"autotune: P = {periodicity} | valid mae = {val_mae}, reject!")

        context_len = convert_context_len(1000, 300, best_valid_p)
        train = [x[-context_len:] for x in data_train]
        mae = evaluate_mae(model, train, data_test, batch_size, device, best_valid_p)
        mae = np.round(mae, 2)
        print(f"dataset = {dataset}, MAE = {mae:.2f}")

