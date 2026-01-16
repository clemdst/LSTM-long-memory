#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import numpy as np
import torch

def create_dataset(y, look_back=1):
    """Create dataset for time series prediction.
    
    Args:
        y: Input data of shape (T, feature_dim) or (T, 1)
        look_back: Number of time steps to look back
        
    Returns:
        X, Y: Input and target sequences
    """
    data_yp, data_yc = [], []
    
    # Handle both 1D and multi-dimensional inputs
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    for i in range(len(y) - look_back):
        data_yp.append(y[i])      # Keep all features
        data_yc.append(y[i + look_back])  # Keep all features
    
    return np.array(data_yp), np.array(data_yc)

def padding(data, length, input_size):
    term = [0] * input_size
    data_pad = []
    for text in data:
        if len(text) >= length:
            text = np.array(text[0:length])
        else:
            pad_list = np.array([term] * (length - len(text)))
            text = np.vstack([np.array(text), pad_list])
        data_pad.append(text)
    data_pad = np.array(data_pad)
    return data_pad

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# transform data to tensor in torch
def to_torch(state):
    state = torch.from_numpy(state).float()
    return state



def unfold_sequence_to_supervised_dataset(data, seq_length, forecast_horizon=1, device='cpu'):
    """
    Create a supervised dataset from multivariate time series sequence.
    with a rolling window approach.
    Args:
        data: torch.Tensor of shape (k, T)
        seq_length: length of the input window (size of X)
        forecast_horizon: number of steps to predict (size of y)

    Returns:
        X: (n_samples, seq_length, k)
        y: (n_samples, forecast_horizon, k)
    """
    data = data.to(device).T  # (T, k) on specified device
    T, k = data.shape

    n_samples = T - seq_length - forecast_horizon + 1

    X = torch.zeros((n_samples, seq_length, k), device=device)
    y = torch.zeros((n_samples, forecast_horizon, k), device=device)

    for idx in range(n_samples):
        X[idx] = data[idx:idx+seq_length]
        y[idx] = data[idx+seq_length:idx+seq_length+forecast_horizon]

    return X.float(), y.float()