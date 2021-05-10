import numpy as np
import pandas as pd
import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pickle
import shutil
from Constants import *
from Custom_Dataloader import *
from Encoder import *
from Decoder import *
from Classifier import *


# Instantiating dataloaders

# In the following two lines of code, replace "formatted_train.csv" and "formatted_test.csv" with None when you are running the code second time
# In the first run of this code block it creates two directories --- "train" and "test", which are reusable in subsequent runs if filenames are replaced with None
train_dataset = CustomDataset(csv_file="formatted_train.csv", directory_name="train", datapoints=np.arange(500000, 700000), train_mode=True)
val_dataset = CustomDataset(csv_file="formatted_train.csv", directory_name="val", datapoints=np.arange(700000, 800000), train_mode=False, vocabulary_location="train")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.generate_batch)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=val_dataset.generate_batch)
num_batches = len(train_dataloader)


# function used for phase 1 training and validation
# phase 1 correspond to the training and validation for the autoencoder branch

def train_and_validate1(sequences, encoder, decoder, criterion, encoder_optimizer=None, decoder_optimizer=None, train=True):

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    sequences = sequences.to(device)
    
    if train:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    sequence_length = sequences.size(0)
    encoder_outputs, encoder_hidden = encoder(sequences)
    encoder_outputs = encoder_outputs.to(device)
    encoder_hidden = (encoder_hidden[0].to(device), encoder_hidden[1].to(device))
    p = vocabulary.stoi["<pad>"]
    loss = 0

    decoder_input = torch.tensor([vocabulary.stoi["<start>"]]*sequences.size(1), device=device)
    (hidden_state, cell_state) = None, None
    
    for t in range(1, sequence_length):
        decoder_output, (hidden_state, cell_state) = decoder(decoder_input, encoder_outputs, (hidden_state, cell_state), encoder_hidden)
        s = sequences[t, :]
        # loss += criterion(decoder_output, sequences[t,:])
        decoder_input = s.to(device)  # Teacher forcing
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
        encoder_hidden = None
        mask1 = (s!=p)
        s = s[mask1]
        loss += criterion(decoder_output[:s.size(0), :], s)

    if train:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item() / float(sequence_length)
  
  
# function used for phase 2 training and validation
# phase 2 corresponds to the training and validation for the classifier branch

def train_and_validate2(sequences, labels, encoder, classifier, criterion, encoder_optimizer=None, classifier_optimizer=None, train=True):

    encoder = encoder.to(device)
    classifier = classifier.to(device)
    
    if train:
        encoder_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

    encoder_outputs, _ = encoder(sequences.to(device))
    encoder_outputs = encoder_outputs.to(device)
    labels = labels.to(device)

    classifier_output = classifier(encoder_outputs)
    _, binarized_output = torch.max(classifier_output, 1)
    correct_count = torch.sum(binarized_output==labels.data)
    loss = criterion(classifier_output, labels)

    if train:
        loss.backward()
        encoder_optimizer.step()
        classifier_optimizer.step()

    return loss.item() / float(sequences.size(1)), correct_count
  
  
# function used for overall training and validation
# training is done in an alternate fashion
# in each epoch we first train the autoencoder and then train the classifier along with the encoder

def train_and_validate(encoder, decoder, classifier, epochs):
    losses1 = []
    val_losses1 = []
    losses2 = []
    val_losses2 = []

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    classifier = classifier.to(device)
    
    encoder_optimizer_for_autoencoder = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE_FOR_ENCODER_IN_AUTOENCODER)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE_FOR_DECODER)
    encoder_optimizer_for_classifier = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE_FOR_ENCODER_IN_CLASSIFIER)
    classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE_FOR_CLASSIFIER)
    
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    train_steps = len(train_dataloader)
    validation_steps = len(val_dataloader)

    for epoch in range(1, epochs + 1):

        print(f"Epoch {epoch}/{epochs}:", "-"*len(f"Epoch {epoch}/{epochs}:"), sep="\n")

        # Phase 1 training

        encoder.train()
        decoder.train()
        initial_time = time.time()
        loss1 = 0.0
        for i, (sequences, _) in enumerate(train_dataloader):
            t = time.time()
            l = train_and_validate1(sequences, encoder, decoder, criterion1, encoder_optimizer_for_autoencoder, decoder_optimizer)
            loss1 += l
            print(f"\r\tPhase 1: [{i + 1: 4d}/{train_steps}] |{'='*int(45/train_steps * (i + 1))}{'>' if int(45/train_steps * (i + 1))!=45 else ''}{' '*max(44 - int(45/train_steps * (i + 1)), 0)}|  Loss = {l:.7f}  Time left = {(time.time() - t)*(train_steps - i - 1):.2f} sec", end="")
            
        loss1 = loss1/float(train_steps)
        total_train_time = time.time() - initial_time
        print(f"\r\tPhase 1: Average train loss = {loss1:.7f}  Total train time = {total_train_time:.2f} sec", end="")
        losses1.append(loss1)

        # Phase 1 validation

        encoder.eval()
        decoder.eval()
        val_loss1 = 0.0
        for i, (sequences, _) in enumerate(val_dataloader):
            l = train_and_validate1(sequences, encoder, decoder, criterion1, train=False)
            val_loss1 += l
            print(f"\r\tPhase 1: Average train loss = {loss1:.7f}  Total train time = {total_train_time:.2f} sec  |  Average validation loss = {'.'*(i%4)}", end="")
            
        val_loss1 = val_loss1/float(validation_steps)
        print(f"\r\tPhase 1: Average train loss = {loss1:.7f}  Total train time = {total_train_time:.2f} sec  |  Average validation loss = {val_loss1:.7f}")
        val_losses1.append(val_loss1)

        # Phase 2 training

        encoder.train()
        classifier.train()
        initial_time = time.time()
        train_correct_count = 0
        loss2 = 0.0
        for i, (sequences, labels) in enumerate(train_dataloader):
            t = time.time()
            l, c = train_and_validate2(sequences, labels, encoder, classifier, criterion2, encoder_optimizer_for_classifier, classifier_optimizer)
            loss2 += l
            train_correct_count += c
            print(f"\r\tPhase 2: [{i + 1: 4d}/{train_steps}] |{'='*int(45/train_steps * (i + 1))}{'>' if int(45/train_steps * (i + 1))!=45 else ''}{' '*max(44 - int(45/train_steps * (i + 1)), 0)}|  Loss = {l:.7f}  Time left = {(time.time() - t)*(train_steps - i - 1):.2f} sec", end="")
            
        loss2 = loss2/float(train_steps)
        train_accuracy = float(train_correct_count)/float(train_steps*BATCH_SIZE) * 100
        total_train_time = time.time() - initial_time
        print(f"\r\tPhase 2: Average train loss = {loss2:.7f}  Total train time = {total_train_time:.2f} sec", end="")
        losses2.append(loss2)

        # Phase 2 validation
        encoder.eval()
        classifier.eval()
        val_correct_count = 0
        val_loss2 = 0.0
        for i, (sequences, labels) in enumerate(val_dataloader):
            l, c = train_and_validate2(sequences, labels, encoder, classifier, criterion2, train=False)
            val_loss2 += l
            val_correct_count += c
            print(f"\r\tPhase 2: Average train loss = {loss2:.7f}  Total train time = {total_train_time:.2f} sec  |  Average validation loss = {'.'*(i%4)}", end="")
            
        val_loss2 = val_loss2/float(validation_steps)
        val_accuracy = float(val_correct_count)/float(validation_steps*BATCH_SIZE) * 100
        print(f"\r\tPhase 2: Average train loss = {loss2:.7f}  Total train time = {total_train_time:.2f} sec  |  Average validation loss = {val_loss2:.7f}")
        print(f"\tTraining accuracy = {train_accuracy:.4f} %\tValidation accuracy = {val_accuracy:.4f} %")
        val_losses2.append(val_loss2)

    # Plotting losses of each phase
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), dpi=150)
    axes[0].plot(range(1, epochs + 1), losses1, color="blue", label="Training loss")
    axes[0].plot(range(1, epochs + 1), val_losses1, color="orange", label="Validation loss")
    axes[1].plot(range(1, epochs + 1), losses2, color="blue", label="Training loss")
    axes[1].plot(range(1, epochs + 1), val_losses2, color="orange", label="Validation loss")
    axes[0].set_title("Training and Validation Loss for autoencoder")
    axes[1].set_title("Training and validation Loss for classifier")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Loss")
    plt.savefig("loss_plot.png")
    fig.show()
                  
                  
encoder = Encoder(VOCABULARY_SIZE, EMBEDDING_DIM, ENCODER_HIDDEN_SIZE, ENCODER_LSTM_LAYERS, ATTENTION_UNITS)
decoder = Decoder(VOCABULARY_SIZE, EMBEDDING_DIM, ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, DECODER_LSTM_LAYERS)
classifier = Classifier(dense_layer_units=CLASSIFIER_DENSE_LAYER_UNITS, encoder_output_dim=ENCODER_HIDDEN_SIZE)
# train and validate
train_and_validate(encoder, decoder, classifier, EPOCHS)
                  

# save trained models

saved_models_dir = "saved_models"
if not os.path.exists(saved_models_dir):
    os.mkdir(saved_models_dir)
DECODER_PATH = saved_models_dir + "/decoder.pt"
ENCODER_PATH = saved_models_dir + "/encoder.pt"
CLASSIFIER_PATH = saved_models_dir + "/classifier.pt"

if os.path.exists(DECODER_PATH):
    os.remove(DECODER_PATH)
torch.save(decoder, DECODER_PATH)

if os.path.exists(ENCODER_PATH):
    os.remove(ENCODER_PATH)
torch.save(encoder, ENCODER_PATH)

if os.path.exists(CLASSIFIER_PATH):
    os.remove(CLASSIFIER_PATH)
torch.save(classifier, CLASSIFIER_PATH)
