import torch
import torch.nn as nn
import numpy as np

class LSTM_FC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device):
        super(LSTM_FC, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        print('************** Model ****************')
        print('Neural Network:\tLSTM')
        print(f'Input Size:\t{self.input_size}')
        print(f'Hidden Size:\t[{self.hidden_size[0]},{self.hidden_size[1]}]')
        print(f'No. of Layers:\t{self.num_layers}')
        print(f'Batch Size:\t{self.batch_size}')
        print('*************************************')

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0], num_layers=self.num_layers, batch_first=True, dropout=0.05 if num_layers > 1 else 0.0).to(self.device)  
        self.fc1 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1], bias=True).to(self.device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.hidden_size[1], out_features=2, bias=True).to(self.device)

    def forward(self, x, lengths):

        if x.size(0) == 0 or lengths is None or lengths.numel() == 0:  
            return torch.empty(0, 2, device=self.device) 

        valid_seq_mask = lengths > 0
        if not valid_seq_mask.any(): 
            return torch.zeros(x.size(0), 2, device=self.device)  
            
        x_valid = x[valid_seq_mask]
        lengths_valid = lengths[valid_seq_mask]

        lengths_sorted, sorted_indices = lengths_valid.sort(descending=True)
        x_sorted = x_valid[sorted_indices]

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True)

        _, (h_n, _) = self.lstm(x_packed)

        lstm_final_output = h_n[-1, :, :]  

        x_fc1 = self.fc1(lstm_final_output)
        x_relu = self.relu(x_fc1)
        x_final_prediction = self.fc2(x_relu) 

        _, original_indices_valid = sorted_indices.sort()
        reordered_predictions = x_final_prediction[original_indices_valid]

        full_batch_output = torch.zeros(len(lengths), 2, device=self.device) 

        full_batch_output[valid_seq_mask] = reordered_predictions
        return full_batch_output
