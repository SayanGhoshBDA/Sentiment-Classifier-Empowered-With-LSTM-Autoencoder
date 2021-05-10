import numpy as np
import pandas as pd
import os
import time
import torch
from Constants import *
from Encoder import *
from Decoder import *
from Classifier import *
from Custom_Dataloader import *


def evaluate_reconstruction(encoder_path, decoder_path, examples):
    encoder = torch.load(encoder_path).to(device)
    decoder = torch.load(decoder_path).to(device)
    sequence_length = examples.size(0)
    encoder_outputs, _ = encoder(examples.to(device))
    encoder_outputs = encoder_outputs.to(device)

    final_output = torch.tensor([], dtype=torch.long)

    decoder_input = torch.tensor([vocabulary.stoi["<start>"]]*examples.size(1), device=device)
    (hidden_state, cell_state) = [torch.zeros(1, examples.size(1), decoder.hidden_size).to(device), torch.zeros(1, examples.size(1), decoder.hidden_size).to(device)]
    criterion = nn.CrossEntropyLoss()
    loss = 0
    for t in range(1, sequence_length):
        decoder_output, (hidden_state, cell_state) = decoder(decoder_input, (hidden_state, cell_state), encoder_outputs)
        _, topi = decoder_output.topk(1)
        for j in range(examples.size(1)):
            if examples[t, j].item()==vocabulary.stoi["<end>"]:
                break
            loss += criterion(decoder_output[j, :].unsqueeze(0), examples[t, j].unsqueeze(0).to(device))
        final_output = torch.cat((final_output, topi.to(torch.device('cpu'))), dim=1)
        decoder_input = examples[t, :].squeeze().to(device)  # Teacher forcing
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
    
    for i in range(examples.size(1)):
        print(f"{i}>\nOriginal example:", " ".join(vocabulary.itos[y] for y in examples[:, i]))
        print("\nReconstructed example:", " ".join(vocabulary.itos[y] for y in final_output[i, :]))
        print("\n")
        
        
test_dataset = CustomDataset(csv_file="formatted_test.csv", directory_name="test", datapoints=np.arange(100000, 300000), train_mode=False, vocabulary_location="train")
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True, collate_fn=test_dataset.generate_batch)
evaluate_reconstruction("saved_models/encoder.pt", "saved_models/decoder.pt", next(iter(test_dataloader))[0])


def evaluate_classification(encoder_path, classifier_path, dataloader=None, examples=None):
    '''
    returns accuracy, precision, recall
    '''
    encoder = torch.load(encoder_path).to(device)
    classifier = torch.load(classifier_path).to(device)
    def check_performance_for_batch(batch):
        sequences, labels = batch
        sequence_length = sequences.size(1)

        encoder_outputs, _ = encoder(sequences.to(device))
        encoder_outputs = encoder_outputs.to(device)
        labels = labels.to(device)

        classifier_output = classifier(encoder_outputs)
        _, binarized_output = torch.max(classifier_output, 1)
        correct_count = torch.sum(binarized_output==labels).item()
        true_positives = torch.sum(binarized_output*labels).item()
        x = torch.bincount(binarized_output + labels*2)
        x = torch.cat((x.cpu(), torch.tensor([0]*(4 - x.size(0)))))
        false_positives = x[1].item()
        false_negatives = x[2].item()

        return correct_count, true_positives, false_positives, false_negatives, sequence_length
      
    if dataloader is not None:
        correct_count = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        num_examples = 0
        for batch in dataloader:
            cc, tp, fp, fn, sl = check_performance_for_batch(batch)
            correct_count += cc
            true_positives += tp
            false_positives += fp
            false_negatives += fn
            num_examples += sl
        return float(correct_count)/float(num_examples) * 100, float(true_positives)/float(true_positives + false_positives + 1e-21) * 100, float(true_positives)/float(true_positives + false_negatives + 1e-21) * 100
    if examples is not None:
        cc, tp, fp, fn, sl = check_performance_for_batch(examples)
        return float(cc)/float(sl) * 100, float(tp)/float(tp + fp + 1e-21) * 100, float(tp)/float(tp + fn + 1e-21) * 100
   
  
test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=True, collate_fn=test_dataset.generate_batch)
evaluate_classification("saved_models/encoder.pt", "saved_models/classifier.pt", test_dataloader)
