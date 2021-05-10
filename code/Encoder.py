import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from Constants import *


# self-attention mechanism used inside the encoder

class IntraAttention(nn.Module):
    def __init__(self, query_or_key_or_value_dim, units, return_sequences):
        super(IntraAttention, self).__init__()
        self.units = units
        self.W1 = nn.Linear(in_features=query_or_key_or_value_dim, out_features=units)
        self.W2 = nn.Linear(in_features=query_or_key_or_value_dim, out_features=units)
        self.V = nn.Linear(in_features=units, out_features=1)
        self.return_sequences = return_sequences
    
    def forward(self, query_or_key_or_value):
        intermediate_state = F.tanh(torch.add(self.W1(query_or_key_or_value),self.W2(query_or_key_or_value)))
        attention_score = self.V(intermediate_state)
        attention_weights = F.softmax(attention_score, dim=0)
        weighted_values = query_or_key_or_value*attention_weights
        if self.return_sequences:
            return weighted_values
        weighted_sum = torch.sum(weighted_values, axis=0)
        return weighted_sum
      
      
# Integrate self-attention mechanism with previous lstm layer inside encoder

class Combo(nn.Module):
    def __init__(self, input_dim, hidden_size, Attention_units):
        super(Combo, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.Attention_units = Attention_units

        self.LSTM_layer = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_size, num_layers = 1, batch_first = False,
                                      dropout = 0.3, bidirectional = False)
        self.Attention_layer = IntraAttention(self.hidden_size, self.Attention_units, return_sequences=True)

    def forward(self, input, hidden_state=None):
        if hidden_state is None:
            output, hidden_state = self.LSTM_layer(input) 
        else:
            output, hidden_state = self.LSTM_layer(input, hidden_state)
        output = self.Attention_layer(output)
        return output, hidden_state
      
      
# Encoder block of autoencoder

class Encoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_size, num_lstm_layers, attention_units):
        super(Encoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.attention_units = attention_units
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        
        input_dims = [embedding_dim] + [hidden_size]*(num_lstm_layers - 1)
        if self.num_lstm_layers>1:
            self.combo_count = 0
            for input_dim, attention_unit in zip(input_dims[:-1], attention_units):
                self.combo_count += 1
                exec(f"self.combo{self.combo_count} = Combo(input_dim, hidden_size, attention_unit)")
        self.lstm = nn.LSTM(input_dims[-1], hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=False)

    def forward(self, x):
        output = self.embedding(x)
        if self.num_lstm_layers>1:
            output, hidden_state = self.combo1(output)
            for i in range(1, self.combo_count):
                exec(f"output, hidden_state = self.combo{i + 1}(output, hidden_state)")
        output, hidden_state = self.lstm(output, hidden_state)
        return output, hidden_state
