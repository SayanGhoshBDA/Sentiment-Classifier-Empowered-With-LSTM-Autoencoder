import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from Constants import *


# attention mechanism used at the time of generating each word from the decoder
# it uses the hidden states of a particular time step of decoder as query and 
# encoder outputs as keys and values

class InterAttention(nn.Module):
    def __init__(self, query_dim, key_or_value_dim, units):
        super(InterAttention, self).__init__()
        self.units = units
        self.W1 = nn.Linear(in_features=query_dim, out_features=units)
        self.W2 = nn.Linear(in_features=key_or_value_dim, out_features=units)
        self.V = nn.Linear(in_features=units, out_features=1)
    
    def forward(self, query, key_or_value):
        intermediate_state = F.tanh(torch.add(self.W1(query), self.W2(key_or_value)))
        attention_score = self.V(intermediate_state)
        attention_weights = F.softmax(attention_score, dim=0)
        weighted_values = key_or_value*attention_weights
        weighted_sum = torch.sum(weighted_values, axis=0)
        return weighted_sum
      
      
# decoder block of the autoencoder

class Decoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, encoder_output_dim, hidden_size, num_lstm_layers):
        super(Decoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = EMBEDDING_DIM
        self.encoder_output_dim = encoder_output_dim
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.attention_for_cellstate = InterAttention(hidden_size, encoder_output_dim, 20)
        self.attention_for_hiddenstate = InterAttention(hidden_size, encoder_output_dim, 20)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.initial_hiddenstate = nn.Linear(encoder_output_dim, hidden_size)
        self.initial_cellstate = nn.Linear(encoder_output_dim, hidden_size)
        input_dims = [encoder_output_dim*2 + embedding_dim] + [hidden_size]*(num_lstm_layers - 1)
        for lstm_num, input_dim in zip(range(1, self.num_lstm_layers  + 1), input_dims):
            exec(f"self.lstm{lstm_num} = nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=False, dropout=0.2, bidirectional=False)")
        self.fc = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, x, encoder_output, hidden, encoder_hidden=None):
        x = self.embedding(x.unsqueeze(0))
        if encoder_hidden is not None:
            encoder_hiddenstate, encoder_cellstate = encoder_hidden
            encoder_hiddenstate = torch.mean(encoder_hiddenstate, dim=0, keepdim=True)
            encoder_cellstate = torch.mean(encoder_cellstate, dim=0, keepdim=True)
            hiddenstate = self.initial_hiddenstate(encoder_hiddenstate)
            cellstate = self.initial_cellstate(encoder_cellstate)
            hidden = (hiddenstate, cellstate)
        else:
            hiddenstate, cellstate = hidden
        att1 = self.attention_for_cellstate(cellstate, encoder_output)
        att2 = self.attention_for_hiddenstate(hiddenstate, encoder_output)
        output = torch.cat([x, att1.unsqueeze(0), att2.unsqueeze(0)], dim=-1)
        output, hidden = self.lstm1(output, hidden)
        for lstm_num in range(2, self.num_lstm_layers + 1):
            exec(f"output, hidden = self.lstm{lstm_num}(output, hidden)")
        x = self.fc(output).squeeze(0)

        return x, hidden
