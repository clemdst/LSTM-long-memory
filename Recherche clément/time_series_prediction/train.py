#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import create_dataset, to_torch
import time_series_prediction.models as models

import importlib
importlib.reload(models)


def train_model(dataset, 
                algorithm='MLSTMFixD',
                epochs=1000,
                lr=0.01,
                hidden_size=1,
                input_size=1,
                output_size=1,
                K=100,
                num_layers=1,
                embedding_dim=None,
                patience=100,
                train_size=2000,
                validate_size=1200,
                start=0,
                end=1,
                verbose=True):
    """
    Train a time series prediction model on the given dataset.
    
    Parameters:
    -----------
    dataset : array-like
        Time series data to train on
    algorithm : str
        Model type: 'RNN', 'LSTM', 'mRNN_fixD', 'mLSTM_fixD', 'mRNN', 'mLSTM'
    epochs : int
        Maximum number of training epochs
    lr : float
        Learning rate
    hidden_size : int
        Number of hidden units
    input_size : int
        Number of input features (default 1 for time series)
    output_size : int
        Number of output features (default 1 for time series)
    K : int
        Truncate infinite summation at lag K (for memory models)
    num_layers : int
        Number of hidden layers (default 1)
    embedding_dim : int or None
        If not None, add embedding layer to project input to this dimension
    patience : int
        Early stopping patience
    train_size : int
        Number of samples for training
    validate_size : int
        Number of samples for validation
    start : int
        Starting seed for multiple runs
    end : int
        Ending seed for multiple runs
    verbose : bool
        Whether to print training progress
        
    Returns:
    --------
    rmse : float
        Root mean squared error on test set
    test_y_r : array
        True test values
    test_predict_r : array
        Predicted test values
    """
    
    batch_size = 1
    rmse_list = []
    mae_list = []
    
    for i in range(start, end):
        seed = i
        if verbose:
            print('seed ----------------------------------', seed)
        
        # Prepare dataset
        if isinstance(dataset, np.ndarray):
            y = dataset
        else:
            y = np.array(dataset)
        
        # Handle different input shapes
        if len(y.shape) == 1:
            # Univariate time series: (T,) -> (T, 1)
            dataset_arr = y.reshape(-1, 1)
        elif len(y.shape) == 2:
            # Check if it's (T, features) or (features, T)
            if y.shape[0] < y.shape[1]:  # Likely (features, T) - transpose it
                dataset_arr = y.T
                if verbose:
                    print(f"Transposed input from {y.shape} to {dataset_arr.shape}")
            else:  # Already (T, features)
                dataset_arr = y
        else:
            raise ValueError(f"Unexpected dataset shape: {y.shape}")
        
        # Detect actual input dimension from data
        actual_input_size = dataset_arr.shape[1]
        if actual_input_size != input_size and verbose:
            print(f"Warning: input_size={input_size} but data has {actual_input_size} features. Using {actual_input_size}.")
            input_size = actual_input_size
            
        # normalize the dataset (normalize each feature independently)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_arr = scaler.fit_transform(dataset_arr)
        
        # use this function to prepare the dataset for modeling
        X, Y = create_dataset(dataset_arr, look_back=1)

        # split into train and test sets
        train_x, train_y = X[0:train_size], Y[0:train_size]
        validate_x, validate_y = X[train_size:train_size + validate_size], Y[train_size:train_size + validate_size]
        test_x, test_y = X[train_size + validate_size:len(Y)], Y[train_size + validate_size:len(Y)]

        # reshape input to be [time steps, samples, features]
        # X and Y now have shape (T, feature_dim) after create_dataset
        train_x = train_x.reshape(train_x.shape[0], batch_size, -1)
        validate_x = validate_x.reshape(validate_x.shape[0], batch_size, -1)
        test_x = test_x.reshape(test_x.shape[0], batch_size, -1)
        train_y = train_y.reshape(train_y.shape[0], batch_size, -1)
        validate_y = validate_y.reshape(validate_y.shape[0], batch_size, -1)
        test_y = test_y.reshape(test_y.shape[0], batch_size, -1)
        
        # Update input_size and output_size based on actual data shape
        input_size = train_x.shape[2]
        output_size = train_y.shape[2]

        torch.manual_seed(seed)
        
        # Detect device (CUDA if available, otherwise CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose and i == start:
            print(f'Using device: {device}')
            if device.type == 'cuda':
                print(f'GPU: {torch.cuda.get_device_name(0)}')
        
        # initialize model
        if algorithm == 'RNN':
            model = models.RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                             num_layers=num_layers, embedding_dim=embedding_dim)
        elif algorithm == 'LSTM':
            model = models.LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                              num_layers=num_layers, embedding_dim=embedding_dim)
        elif algorithm == 'MRNNFixD':
            model = models.MRNNFixD(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                                  k=K, num_layers=num_layers, embedding_dim=embedding_dim)
        elif algorithm == 'MRNN':
            model = models.MRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                              k=K, num_layers=num_layers, embedding_dim=embedding_dim)
        elif algorithm == 'MLSTMFixD':
            model = models.MLSTMFixD(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                                   k=K, num_layers=num_layers, embedding_dim=embedding_dim)
        elif algorithm == 'MLSTM':
            model = models.MLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                               k=K, num_layers=num_layers, embedding_dim=embedding_dim)
        else:
            raise ValueError(f'Unknown algorithm: {algorithm}')
        
        # Move model to device
        model = model.to(device)
            
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        best_loss = np.infty
        best_train_loss = np.infty
        stop_criterion = 1e-5
        rec = np.zeros((epochs, 3))
        epoch = 0
        val_loss = -1
        train_loss = -1
        cnt = 0

        def train():
            model.train()
            optimizer.zero_grad()
            target = torch.from_numpy(train_y).float().to(device)
            output, hx = model(torch.from_numpy(train_x).float().to(device))
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Validation - IMPORTANT: détacher complètement du graphe d'entraînement
            model.eval()
            with torch.no_grad():
                # Ne pas réutiliser hx de l'entraînement !
                val_y, _ = model(torch.from_numpy(validate_x).float().to(device))
                target_val = torch.from_numpy(validate_y).float().to(device)
                val_loss = criterion(val_y, target_val)
            
            model.train()  # Remettre en mode train
            
            # Détacher les valeurs pour éviter l'accumulation de graphes
            return loss.item(), val_loss.item()

        def detach_hidden_state(hx):
            """Helper function to detach hidden states for all model types"""
            if hx is None:
                return None
            elif isinstance(hx, tuple):
                # Check if tuple contains lists (mLSTM/mRNN) or tensors (LSTM)
                if len(hx) > 0 and isinstance(hx[0], list):
                    # mLSTM/mRNN: tuple of lists
                    return tuple([[h.detach() if h is not None else None for h in hidden_list] for hidden_list in hx])
                else:
                    # Regular LSTM: tuple of tensors
                    return tuple(h.detach() if h is not None else None for h in hx)
            elif isinstance(hx, list):
                # mRNN: list of hidden states
                return [h.detach() if h is not None else None for h in hx]
            else:
                # RNN: single tensor
                return hx.detach()
        
        def compute_test(best_model):
            model = best_model
            model.eval()
            with torch.no_grad():
                train_predict, hx = model(to_torch(train_x).to(device))
                train_predict = train_predict.cpu().detach().numpy()
                
                # Detach hidden state for all model types
                hx = detach_hidden_state(hx)
                
                val_predict, hx = model(to_torch(validate_x).to(device), hx)
                
                # Detach hidden state again for test
                hx = detach_hidden_state(hx)
                
                test_predict, _ = model(to_torch(test_x).to(device), hx)
                test_predict = test_predict.cpu().detach().numpy()
            
            # invert predictions
            test_predict_r = scaler.inverse_transform(test_predict[:, 0, :])
            test_y_r = scaler.inverse_transform(test_y[:, 0, :])
            # calculate error
            test_rmse = math.sqrt(mean_squared_error(test_y_r[:, 0], test_predict_r[:, 0]))
            test_mape = (abs((test_predict_r[:, 0] - test_y_r[:, 0]) / test_y_r[:, 0])).mean()
            test_mae = mean_absolute_error(test_predict_r[:, 0], test_y_r[:, 0])
            return test_rmse, test_mape, test_mae, test_y_r, test_predict_r

        while epoch < epochs:
            _time = time.time()
            train_loss_val, val_loss_val = train()
            if (val_loss_val < best_loss):
                best_loss = val_loss_val
                best_epoch = epoch
                best_model = deepcopy(model)
            # stop_criteria = abs(criterion(val_Y, target_val) - val_loss)
            if ((best_train_loss - train_loss_val) > stop_criterion):
                best_train_loss = train_loss_val
                cnt = 0
            else:
                cnt += 1
            if cnt == patience:
                break
            # save training records
            time_elapsed = time.time()-_time
            rec[epoch, :] = np.array([train_loss_val, val_loss_val, time_elapsed])
            if verbose:
                print("epoch: {:2.0f} train_loss: {:2.5f} val_loss: {:2.5f}  time: {:2.1f}s".format(
                    epoch, train_loss_val, val_loss_val, time_elapsed))
            epoch = epoch + 1
            
            # Libérer la mémoire périodiquement
            if epoch % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # make predictions
        test_rmse, test_mape, test_mae, test_y_r, test_predict_r = compute_test(best_model)

        rmse_list.append(test_rmse)
        mae_list.append(test_mae)
        if verbose:
            print('RMSE:{}'.format(rmse_list))
            print('MAE:{}'.format(mae_list))
    
    # Return results from the last run
    return rmse_list[-1], test_y_r[:, 0], test_predict_r[:, 0], best_model, scaler


def extract_hidden_states(model, dataset, scaler):
    """
    Extract hidden states from the trained model for long memory testing.
    
    Parameters:
    -----------
    model : trained model
        The trained RNN/LSTM/mLSTM model
    dataset : array-like
        Original time series data
    scaler : MinMaxScaler
        Scaler used during training
        
    Returns:
    --------
    hidden_states : np.ndarray
        Hidden states of shape (hidden_size, T) for long memory testing
    """
    # Prepare data
    if isinstance(dataset, np.ndarray):
        y = dataset
    else:
        y = np.array(dataset)
    
    dataset_arr = y.reshape(-1, 1)
    dataset_arr = scaler.transform(dataset_arr)
    
    # Prepare dataset
    X, Y = create_dataset(dataset_arr, look_back=1)
    
    # Reshape to match model input: [time_steps, batch_size=1, input_size=1]
    X = np.reshape(X, (X.shape[0], 1, 1))
    
    model.eval()
    hidden_states_list = []
    
    with torch.no_grad():
        hidden_state = None
        device = next(model.parameters()).device
        
        for t in range(len(X)):
            input_t = torch.from_numpy(X[t:t+1]).float().to(device)
            
            # Forward pass
            output, hidden_state = model(input_t, hidden_state)
            
            # Extract hidden state based on model type
            if isinstance(hidden_state, tuple):  # LSTM returns (h, c)
                h = hidden_state[0]
            else:  # RNN returns just h
                h = hidden_state
            
            # Get the hidden state (handle different shapes)
            if len(h.shape) == 3:  # (num_layers, batch, hidden_size)
                last_layer_hidden = h[-1, 0, :].cpu().numpy()
            elif len(h.shape) == 2:  # (batch, hidden_size)
                last_layer_hidden = h[0, :].cpu().numpy()
            else:  # (hidden_size,)
                last_layer_hidden = h.cpu().numpy()
            
            hidden_states_list.append(last_layer_hidden)
    
    # Convert to (hidden_size, T) format
    hidden_states = np.array(hidden_states_list).T
    
    return hidden_states