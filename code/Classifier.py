import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from Constants import *


# classifier branch which is attached to the tail of encoder

class Classifier(nn.Module):
    def __init__(self, dense_layer_units, encoder_output_dim, dropout=0.2, num_classes=2):
        super(Classifier, self).__init__()

        self.num_classes = num_classes

        if isinstance(dense_layer_units, int):
            self.dense_layer_units = [dense_layer_units]
        else:
            self.dense_layer_units = dense_layer_units

        self.num_dense_layers = len(dense_layer_units)
        
        if not isinstance(dropout, list):
            self.dropout = [dropout]*self.num_dense_layers
        else:
            self.dropout = dropout
        
        self.dense_layers = nn.Sequential()
        input_dims = [encoder_output_dim] + self.dense_layer_units[:-1]
        for i in range(self.num_dense_layers):
            self.dense_layers.add_module(f"Classifier_Dense_Layer_{i+1}", nn.Sequential(nn.Linear(in_features=input_dims[i], out_features=self.dense_layer_units[i]), nn.SELU(), nn.Dropout(p=self.dropout[i])))
        
        self.dense_layers.add_module("classification_layer", nn.Linear(in_features=self.dense_layer_units[-1], out_features=self.num_classes))
    
    def forward(self, encoder_output):
        output = torch.mean(encoder_output, dim=0)
        output = self.dense_layers(output)
        return output
