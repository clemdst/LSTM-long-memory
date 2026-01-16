#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it
# under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import layers


class RNN(nn.Module):
    """RNN model for time series prediction
    
    Args:
        input_size: Dimension of input features (1 for raw time series)
        hidden_size: Dimension of RNN hidden state
        output_size: Dimension of output
        num_layers: Number of stacked RNN layers
        embedding_dim: If not None, add embedding layer to project input_size to embedding_dim
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, embedding_dim=None):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Optional embedding layer to project input to higher dimension
        if embedding_dim is not None:
            self.input_embedding = nn.Linear(input_size, embedding_dim)
            rnn_input_size = embedding_dim
        else:
            self.input_embedding = None
            rnn_input_size = input_size
            
        self.rnn = nn.RNN(rnn_input_size, self.hidden_size, num_layers=num_layers)
        self.hidden2output = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_y, hidden_state=None):
        samples = input_y
        
        # Apply embedding if specified
        if self.input_embedding is not None:
            samples = self.input_embedding(samples)
            
        rnn_out, last_rnn_hidden = self.rnn(samples, hidden_state)
        output = self.hidden2output(rnn_out.view(-1, self.hidden_size))
        return output.view(samples.shape[0], samples.shape[1],
                           self.output_size), \
               last_rnn_hidden


class LSTM(nn.Module):
    """LSTM model for time series prediction
    
    Args:
        input_size: Dimension of input features (1 for raw time series)
        hidden_size: Dimension of LSTM hidden state
        output_size: Dimension of output
        num_layers: Number of stacked LSTM layers
        embedding_dim: If not None, add embedding layer to project input_size to embedding_dim
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, embedding_dim=None):
        super(LSTM, self).__init__()
        self.in_size = input_size
        self.h_size = hidden_size
        self.out_size = output_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Optional embedding layer to project input to higher dimension
        if embedding_dim is not None:
            self.input_embedding = nn.Linear(input_size, embedding_dim)
            lstm_input_size = embedding_dim
        else:
            self.input_embedding = None
            lstm_input_size = input_size
            
        # Use nn.LSTM instead of LSTMCell for multi-layer support
        if num_layers > 1:
            self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers=num_layers)
            self.output = nn.Linear(hidden_size, output_size)
        else:
            self.lstm_cell = nn.LSTMCell(lstm_input_size, hidden_size)
            self.output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.out_size)
        
        # Apply embedding if specified
        if self.input_embedding is not None:
            inputs = self.input_embedding(inputs)
        
        if self.num_layers > 1:
            # Multi-layer LSTM forward pass
            lstm_out, hidden_state = self.lstm(inputs, hidden_state)
            outputs = self.output(lstm_out.view(-1, self.h_size))
            return outputs.view(time_steps, batch_size, self.out_size), hidden_state
        else:
            # Single-layer LSTM with LSTMCell
            if hidden_state is None:
                h_0 = torch.zeros(batch_size, self.h_size)
                c_0 = torch.zeros(batch_size, self.h_size)
                hidden_state = (h_0, c_0)
            else:
                h_0 = hidden_state[0]
                c_0 = hidden_state[1]
            for times in range(time_steps):
                h_0, c_0 = self.lstm_cell(inputs[times, :], (h_0, c_0))
                outputs[times, :] = self.output(h_0)
            return outputs, (h_0, c_0)


class MRNNFixD(nn.Module):
    """mRNN with fixed d for time series prediction
    
    Args:
        input_size: Dimension of input features (1 for raw time series)
        hidden_size: Dimension of mRNN hidden state
        output_size: Dimension of output
        k: Truncation parameter for infinite summation
        bias: Whether to use bias in linear layers
        num_layers: Number of stacked mRNN layers
        embedding_dim: If not None, add embedding layer to project input_size to embedding_dim
    """
    def __init__(self, input_size, hidden_size, output_size, k, bias=True, num_layers=1, embedding_dim=None):
        super(MRNNFixD, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Optional embedding layer
        if embedding_dim is not None:
            self.input_embedding = nn.Linear(input_size, embedding_dim)
            first_layer_input_size = embedding_dim
        else:
            self.input_embedding = None
            first_layer_input_size = input_size
            
        self.b_d = Parameter(torch.Tensor(torch.zeros(1, first_layer_input_size)),
                             requires_grad=True)
        # Stack multiple mRNN cells if num_layers > 1
        self.mrnn_cells = nn.ModuleList([
            layers.MRNNFixDCell(
                first_layer_input_size if i == 0 else hidden_size,
                hidden_size,
                output_size if i == num_layers - 1 else hidden_size,
                k
            ) for i in range(num_layers)
        ])
    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_value):
        weights = torch.ones(self.k, 1, d_value.size(1), dtype=d_value.dtype,
                             device=d_value.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = self.get_ws(d_value[0, hidden].
                                                         view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.size(0)
        
        # Apply embedding if specified
        if self.input_embedding is not None:
            inputs = self.input_embedding(inputs)
            
        self.d_matrix = 0.5 * F.sigmoid(self.b_d)
        weights_d = self.get_wd(self.d_matrix)
        
        # Initialize hidden states for all layers if None
        if hidden_state is None:
            hidden_state = [None] * self.num_layers
        
        for times in range(time_steps):
            layer_input = inputs[times, :]
            for layer_idx in range(self.num_layers):
                outputs, hidden_state[layer_idx] = self.mrnn_cells[layer_idx](
                    layer_input, weights_d, hidden_state[layer_idx]
                )
                layer_input = outputs if layer_idx < self.num_layers - 1 else outputs
        
        return outputs, hidden_state


class MRNN(nn.Module):
    """mRNN with dynamic d for time series prediction
    
    Args:
        input_size: Dimension of input features (1 for raw time series)
        hidden_size: Dimension of mRNN hidden state
        output_size: Dimension of output
        k: Truncation parameter for infinite summation
        bias: Whether to use bias in linear layers
        num_layers: Number of stacked mRNN layers
        embedding_dim: If not None, add embedding layer to project input_size to embedding_dim
    """
    def __init__(self, input_size, hidden_size, output_size, k, bias=True, num_layers=1, embedding_dim=None):
        super(MRNN, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Optional embedding layer
        if embedding_dim is not None:
            self.input_embedding = nn.Linear(input_size, embedding_dim)
            first_layer_input_size = embedding_dim
        else:
            self.input_embedding = None
            first_layer_input_size = input_size
            
        # Stack multiple mRNN cells if num_layers > 1
        self.mrnn_cells = nn.ModuleList([
            layers.MRNNCell(
                first_layer_input_size if i == 0 else hidden_size,
                hidden_size,
                output_size if i == num_layers - 1 else hidden_size,
                k
            ) for i in range(num_layers)
        ])

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)
        outputs = torch.Tensor(time_steps, batch_size, self.output_size)
        
        # Apply embedding if specified
        if self.input_embedding is not None:
            inputs = self.input_embedding(inputs)
            
        # Initialize hidden states for all layers if None
        if hidden_state is None:
            hidden_state = [None] * self.num_layers
        
        for times in range(time_steps):
            layer_input = inputs[times, :]
            for layer_idx in range(self.num_layers):
                layer_output, hidden_state[layer_idx] = self.mrnn_cells[layer_idx](
                    layer_input, hidden_state[layer_idx]
                )
                layer_input = layer_output
            outputs[times, :] = layer_output
        
        return outputs, hidden_state


class MLSTMFixD(nn.Module):
    """mLSTM with fixed d for time series prediction
    
    Args:
        input_size: Dimension of input features (1 for raw time series)
        hidden_size: Dimension of mLSTM hidden state
        k: Truncation parameter for infinite summation
        output_size: Dimension of output
        num_layers: Number of stacked mLSTM layers
        embedding_dim: If not None, add embedding layer to project input_size to embedding_dim
    """
    def __init__(self, input_size, hidden_size, k, output_size, num_layers=1, embedding_dim=None):
        super(MLSTMFixD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Optional embedding layer
        if embedding_dim is not None:
            self.input_embedding = nn.Linear(input_size, embedding_dim)
            first_layer_input_size = embedding_dim
        else:
            self.input_embedding = None
            first_layer_input_size = input_size
            
        self.d_values = Parameter(torch.Tensor(torch.zeros(1, hidden_size)),
                                  requires_grad=True)
        self.output_size = output_size
        # Stack multiple mLSTM cells if num_layers > 1
        self.mlstm_cells = nn.ModuleList([
            layers.MLSTMFixDCell(
                first_layer_input_size if i == 0 else hidden_size,
                hidden_size,
                output_size if i == num_layers - 1 else hidden_size,
                k
            ) for i in range(num_layers)
        ])
        self.sigmoid = nn.Sigmoid()

    def get_w(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_value):
        weights = torch.ones(self.k, 1, d_value.size(1), dtype=d_value.dtype,
                             device=d_value.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = self.get_w(d_value[0, hidden].
                                                        view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, hidden_states=None):
        # Apply embedding if specified
        if self.input_embedding is not None:
            inputs = self.input_embedding(inputs)
            
        # Initialize hidden states for all layers if None
        if hidden_states is None:
            hidden_list = [None] * self.num_layers
            h_c_list = [None] * self.num_layers
        else:
            hidden_list = hidden_states[0] if isinstance(hidden_states[0], list) else [hidden_states[0]]
            h_c_list = hidden_states[1] if isinstance(hidden_states[1], list) else [hidden_states[1]]
        
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        self.d_values_sigmoid = 0.5 * F.sigmoid(self.d_values)
        weights_d = self.get_wd(self.d_values_sigmoid)
        
        for times in range(time_steps):
            layer_input = inputs[times, :]
            for layer_idx in range(self.num_layers):
                layer_output, hidden_list[layer_idx], h_c_list[layer_idx] = self.mlstm_cells[layer_idx](
                    layer_input, hidden_list[layer_idx], h_c_list[layer_idx], weights_d
                )
                layer_input = layer_output
            outputs[times, :] = layer_output
        
        return outputs, (hidden_list, h_c_list)


class MLSTM(nn.Module):
    """mLSTM with dynamic d for time series prediction
    
    Args:
        input_size: Dimension of input features (1 for raw time series)
        hidden_size: Dimension of mLSTM hidden state
        k: Truncation parameter for infinite summation
        output_size: Dimension of output
        num_layers: Number of stacked mLSTM layers
        embedding_dim: If not None, add embedding layer to project input_size to embedding_dim
    """
    def __init__(self, input_size, hidden_size, k, output_size, num_layers=1, embedding_dim=None):
        super(MLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Optional embedding layer
        if embedding_dim is not None:
            self.input_embedding = nn.Linear(input_size, embedding_dim)
            first_layer_input_size = embedding_dim
        else:
            self.input_embedding = None
            first_layer_input_size = input_size
            
        # Stack multiple mLSTM cells if num_layers > 1
        self.mlstm_cells = nn.ModuleList([
            layers.MLSTMCell(
                first_layer_input_size if i == 0 else hidden_size,
                hidden_size,
                k,
                output_size if i == num_layers - 1 else hidden_size
            ) for i in range(num_layers)
        ])

    def forward(self, inputs, hidden_state=None):
        # Apply embedding if specified
        if self.input_embedding is not None:
            inputs = self.input_embedding(inputs)
            
        # Initialize hidden states for all layers if None
        if hidden_state is None:
            hidden_list = [None] * self.num_layers
            h_c_list = [None] * self.num_layers
            d_values_list = [None] * self.num_layers
        else:
            hidden_list = hidden_state[0] if isinstance(hidden_state[0], list) else [hidden_state[0]]
            h_c_list = hidden_state[1] if isinstance(hidden_state[1], list) else [hidden_state[1]]
            d_values_list = hidden_state[2] if isinstance(hidden_state[2], list) else [hidden_state[2]]
        
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        
        for times in range(time_steps):
            layer_input = inputs[times, :]
            for layer_idx in range(self.num_layers):
                layer_output, hidden_list[layer_idx], h_c_list[layer_idx], d_values_list[layer_idx] = \
                    self.mlstm_cells[layer_idx](
                        layer_input, hidden_list[layer_idx], h_c_list[layer_idx], d_values_list[layer_idx]
                    )
                layer_input = layer_output
            outputs[times, :] = layer_output
        
        return outputs, (hidden_list, h_c_list, d_values_list)
