import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device):
        super(LSTM_Attention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        print('************** Model ****************')
        print('Neural Network:\tLSTM + Attention')
        print(f'Input Size:\t{self.input_size}')
        print(f'Hidden Size:\t[{self.hidden_size[0]},{self.hidden_size[1]}]')
        print(f'No. of Layers:\t{self.num_layers}')
        print(f'Batch Size:\t{self.batch_size}')
        print('*************************************')

        
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size[0], num_layers=num_layers, batch_first=True, dropout=0.05 if num_layers > 1 else 0.0).to(self.device)
        self.attention_layer = nn.Linear(hidden_size[0], 1).to(self.device)
        self.fc1 = nn.Linear(hidden_size[0], hidden_size[1]).to(self.device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[1], 2).to(self.device) # 2 classes

    def forward(self, x, lengths, return_attention=False):

        if x.size(0) == 0 or lengths is None or lengths.numel() == 0:
            return torch.empty(0, 2, device=self.device) if not return_attention else (torch.empty(0, 2, device=self.device), torch.empty(0, 0, device=self.device))

        valid_seq_mask = lengths > 0
        if not valid_seq_mask.any():
            output_zeros = torch.zeros(x.size(0), 2, device=self.device)
            return output_zeros if not return_attention else (output_zeros, torch.zeros(x.size(0), x.size(1), device=self.device))

        x_valid = x[valid_seq_mask]
        lengths_valid = lengths[valid_seq_mask]

        lengths_sorted, sorted_indices = lengths_valid.sort(descending=True)
        x_sorted = x_valid[sorted_indices]

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True)

        lstm_out_packed, (h_n, c_n) = self.lstm(x_packed)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)

        #################
        max_len_valid = lstm_out.size(1)
        mask = torch.arange(max_len_valid, device=lengths_sorted.device).unsqueeze(0) < lengths_sorted.unsqueeze(1)
        mask = mask.unsqueeze(2).float() # (batch_size_valid, seq_len_max_valid, 1)

        attention_scores = self.attention_layer(lstm_out)
        attention_scores_masked = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores_masked, dim=1)

        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        #################

        x_fc1 = self.fc1(context_vector)
        x_relu = self.relu(x_fc1)
        x_final_prediction = self.fc2(x_relu) # (batch_size_valid, 2)

        full_batch_output = torch.zeros(len(lengths), 2, device=self.device)
        _, original_indices_valid = sorted_indices.sort()
        reordered_predictions = x_final_prediction[original_indices_valid]
        full_batch_output[valid_seq_mask] = reordered_predictions

        if return_attention:
            full_attention_weights = torch.zeros(len(lengths), x.size(1), device=self.device)
            att_reordered = attention_weights.squeeze(-1)[original_indices_valid]
            
            for i_batch, idx_full in enumerate(torch.where(valid_seq_mask)[0]):
                seq_len = lengths_valid[sorted_indices[i_batch]]
                full_attention_weights[idx_full, :seq_len] = att_reordered[i_batch, :seq_len]
                
            return full_batch_output, full_attention_weights

        return full_batch_output