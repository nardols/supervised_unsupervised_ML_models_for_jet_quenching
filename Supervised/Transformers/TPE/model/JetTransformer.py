# inspired by 
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch?utm_source=google&utm_medium=paid_search&utm_campaignid=21057859163&utm_adgroupid=157296750377&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=733936255649&utm_targetid=dsa-2264919291989&utm_loc_interest_ms=&utm_loc_physical_ms=9198796&utm_content=ps-other~latam-en~dsa~tofu~tutorial-artificial-intelligence&accountid=9624585688&utm_campaign=230119_1-ps-other~dsa~tofu_2-b2c_3-latam-en_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&gad_source=1&gad_campaignid=21057859163&gbraid=0AAAAADQ9WsFoKBgsjT7V9T3uUzsQDCCWv&gclid=Cj0KCQjw0LDBBhCnARIsAMpYlAofG9bobYSAEVk8Q5JLj25MLHOrVtXrb0CvOHXXVJjIPwj3GQ605o0aAu_mEALw_wcB

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        # Initialize dimensions
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None, return_attention=False):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # If necessary, aplly mask (useful for preventing attention to certain parts, ex: padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Aplly softmaz to obatin attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Obtain the final output
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)

        #Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_probs



class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)      # two fully connected linear layers with input and output dimensions defined as d_model and d_ff
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model) # a tensor filled with zeros, which will be populated with positional encondings
        position = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1) # a tensor that contains the position indices for each position in the sequence
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * - (math.log(10000.0) / d_model)) # used to scale the position indices in a specific way

        # The sine function is applied for even indices and the cosine to odd indices of pe
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # pe is regirested as a buffer: it will be part of the module's state but it won't be considered a trainable parameter
        self.register_buffer('pe', pe.unsqueeze(0))

    # Adds the positional encodings to the input x
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) # Layer normalization, applied to smooth the layer's input
        self.dropout = nn.Dropout(dropout) # Prevent overfitting

    def forward(self, x, mask, return_attention=False):
        attn_output, attn_probs = self.self_attn(x, x, x, mask) # The input x is passed through the multi-head self-attention mechanism
        x = self.norm1(x + self.dropout(attn_output)) # Attention output is added to the original input, followed by dropout and normalization using norm1
        ff_output = self.feed_forward(x) # The output from the previous step is passed through the position-wise feed-forward network
        x = self.norm2(x + self.dropout(ff_output)) # The feed-forward output is added to the input of this stage, followed by dropout and normalization using norm2
        if return_attention:
            return x, attn_probs
        return x


"""

***
THE DECODER IS NECESSARY TO GENERATE A SEQUENCE, BUT IN THIS CASE WE USE THE TRANSFORMER ONLY FOR BINARY CLASSIFICATION AND REGRESSION
***

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads # Multi-head self-attention mechanism for the target sequence
        self.cross_attn = MultiHeadAttention(d_model, num_heads) # Multi-head attention mechanism that attends to the encoder's output
        self.feed_forward = PositionalWiseFeedForward(d_model, d_ff) # Position-wise feed-forward neural network
        # Layer normalization components
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) # dropout layer for regularization


    def forward(self, x, enc_output, src_mark, tgt_mask):
        # enc_output: the output for the corresponding encoder
        # src_mask: source mask to ignore certain parts of the encoder's output
        # tgt_mask: target mask to ignore certain parts of the decoder's input
        attn_output = self.self_attn(x, x, x, tgt_mask) # input x is processed through a self-attention mechanism
        x = self.norm1(x + self.dropout(attn_output)) # output from self_attn is added to the original x, followed by dropout and normalization using norm1
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask) # the normalized output from the previous step is processed through a cross-attention mechanism that attends to the encoder's output enc_output
        x = self.norm2(x + self.dropout(attn_output)) # output from cross-attention is added to the input of this stage, followed by dropout and normalization using norm2
        ff_output = self.feed_forward(x) # output from previous step is passed through the feed-forward network
        x = self.norm3(x + self.dropout(ff_output)) # the feed-forward output is added to the input of this stage, followed by dropout and normalization using norm3
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
"""

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_vector = nn.Parameter(torch.randn(hidden_dim)) 

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        # Calculate the attention score for each step
        attn_scores = torch.matmul(x, self.attention_vector)  # (batch_size, seq_len)

        # softmax
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)

        # weighted mean
        pooled_output = torch.sum(attn_weights * x, dim=1)  # (batch_size, hidden_dim)

        return pooled_output
        

class JetTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_dim, num_classes, dropout, task, max_seq_length):
        
        """
        task: classification or regression
        """
        
        super(JetTransformer, self).__init__()
        self.task = task

        self.pooling = AttentionPooling(model_dim)

        self.input_fc = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

        if task == 'classification':
            self.output_fc = nn.Linear(model_dim, num_classes) 
        elif task == 'regression':
            self.output_fc = nn.Linear(model_dim, 1)
        else:
            print('Invalid choice of task')

    def forward(self, x, mask=None, return_attention=False):
        x = self.input_fc(x)
        x = self.positional_encoding(x)
        all_attn = []
        for layer in self.encoder_layers:
            if return_attention:
                x, attn_probs = layer(x, mask, return_attention=True)
                all_attn.append(attn_probs)
            else:
                x = layer(x, mask)
        pooled_output = self.pooling(x)
        output = self.output_fc(self.dropout(pooled_output))
        if self.task == 'classification':
            if return_attention:
                return output, all_attn
            return output.squeeze(-1)
        elif self.task == 'regression':
            if return_attention:
                return output.squeeze(-1), all_attn
            return output.squeeze(-1)
