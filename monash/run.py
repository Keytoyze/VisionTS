import sys
sys.path.append("../")
import numpy as np
import torch
import math
import einops
import gluonts_util
from visionts import VisionTS
from typing import NamedTuple

DATASETS_PERIODICITY = {
    "m1_monthly": 12,
    "monash_m3_monthly": 6,
    "monash_m3_other": 2,
    "m4_monthly": 6,
    "m4_weekly": 1,
    "m4_daily": 1,
    "m4_hourly": 24,
    "tourism_quarterly": 4, 
    "tourism_monthly": 12, 
    "cif_2016_6": 3,
    "cif_2016_12": 6,
    "australian_electricity_demand": 336, 
    "bitcoin": 1,
    "pedestrian_counts": 48, 
    "vehicle_trips_without_missing": 7,
    "kdd_cup_2018_without_missing": 96,
    "weather": 365,
    "nn5_daily_without_missing": 7,
    "nn5_weekly": 1, 
    "car_parts_without_missing": 6,
    "fred_md": 1, 
    "traffic_hourly": 24,
    "traffic_weekly": 1, 
    "rideshare_without_missing": 1,
    "hospital": 12, 
    "covid_deaths": 1, 
    "temperature_rain_with_missing": 365,
    "sunspot_with_missing": 30,
    "saugeenday": 30,
    "us_births": 365
}

class MetaData(NamedTuple):
    freq: str
    target_dim: int
    prediction_length: int
    feat_dynamic_real_dim: int = 0
    past_feat_dynamic_real_dim: int = 0
    split: str = "test"

def compute_mae_with_nan(y_true, y_pred):
    mae = []
    for i in range(y_true.size(0)):
        diff = torch.abs(y_pred[i, :, 0] - y_true[i, :, 0])
        if torch.isnan(diff).all():
            continue
        diff = diff[~torch.isnan(diff)].mean().item()
        mae.append(diff)
    return np.asarray(mae)

def process_nan(array):
    not_nan_mask = ~np.isnan(array)
    if not not_nan_mask.any():
        return np.zeros_like(array)
    indices = np.arange(array.size)
    try:
        interpolated_array = np.interp(indices, indices[not_nan_mask], array[not_nan_mask])
    except:
        breakpoint()
    return interpolated_array

if __name__ == "__main__":

    device = 'cuda:0'
    batch_size = 1024

    model = VisionTS(ckpt_dir="../ckpt/").to(device)

    seq_len = 1000
    norm_mae_list = []
    for dataset_i, dataset in enumerate(DATASETS_PERIODICITY):
        # load gluonts data
        test_data, metadata = gluonts_util.get_gluonts_test_dataset(dataset)
        data_train = [x['target'] for x in test_data.input]
        data_test = [x['target'] for x in test_data.label]

        periodicity = DATASETS_PERIODICITY[dataset]
    
        if periodicity == 1:
            context_len = 300
        else:
            context_len = int(round(1000 / periodicity)) * periodicity
        train = [process_nan(x[-context_len:]) for x in data_train]

        # We combine testing data with the context lengths
        seq_len_to_group_data = {}
        for i in range(len(train)):
            train_len = len(train[i])
            if train_len not in seq_len_to_group_data:
                seq_len_to_group_data[train_len] = [[], []]
            seq_len_to_group_data[train_len][0].append(train[i])
            seq_len_to_group_data[train_len][1].append(data_test[i])
        
        mae_list = []
        naive_mae_list = []
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

            last_values = []
            for i in range(len(cur_train)):
                cnt = 0
                while True:
                    last_value = cur_train[i, -1 - cnt]
                    if not torch.isnan(last_value):
                        break
                    cnt += 1
                last_values.append(last_value)
            naive_pred = einops.repeat(torch.stack(last_values).to(device), 'b 1 -> b t 1', t=cur_test.shape[1])
            naive_mae = compute_mae_with_nan(naive_pred, cur_test)
            naive_mae_list.append(naive_mae)
            
        mae = np.mean(np.concatenate(mae_list))
        naive_mae = np.mean(np.concatenate(naive_mae_list))
        norm_mae_list.append(mae / naive_mae)
        mae = np.round(mae, 2)
        naive_mae = np.round(naive_mae, 2)
        print(f"dataset = {dataset}, MAE = {mae:.2f}, {naive_mae}")




