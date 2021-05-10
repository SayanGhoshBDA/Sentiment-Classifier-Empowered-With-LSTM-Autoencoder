import os
import pandas as pd
import shutil
import numpy as np
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
import pickle
import re
import string
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from Constants import *


class CustomDataset(Dataset):
    def __init__(self, csv_file = None, directory_name = "train", datapoints=range(1000000), train_mode=True, vocabulary_location=None):
        '''
        'csv_file' is the .csv file with reviews and labels

        'directory_name' is the name of the directory where you want to save the formatted data

        'train_mode' says whether the creaetd dataseet is being used for training purpose or not

        'vocabulary_location' is the name of the directory where the created vocabulary will be kept,
        if None, it is same as 'directory_name'
        '''
        
        self.directory_name = directory_name
        if vocabulary_location is None:
            vocabulary_location = directory_name
        global vocabulary

        # create equi dimensional tensors for each example and save as file along with integer class label seperated by tab
        if csv_file is not None:
            data = pd.read_csv(csv_file, header=0, names=["review", "label"]).iloc[datapoints, :].reset_index(drop=True)
            if os.path.isdir(directory_name):
                shutil.rmtree(directory_name)
            os.mkdir(directory_name)
            self.num_examples = len(data.index)
            self.num_sub_directories = int(np.sqrt(self.num_examples)) + 1
            self.num_examples_per_sub_directory = self.num_examples//(self.num_sub_directories - 1)
            
            reviews = list(map(self.format, data["review"]))
            self.tokenizer = get_tokenizer(tokenizer="basic_english", language="en") 
            if train_mode:
                counter = Counter()
                for example in reviews:
                    counter.update(self.tokenizer(example))
                vocabulary = Vocab(counter, max_size=None, min_freq=5, specials=["<unk>", "<pad>", "<start>", "<end>"])
                with open(f"{vocabulary_location}/vocabulary.pickle", "wb") as f:
                    pickle.dump(vocabulary, f)
            
            for i in range(self.num_examples):
                if i%self.num_examples_per_sub_directory==0:
                    sub_directory = f"{directory_name}/{i//self.num_examples_per_sub_directory}"
                    os.mkdir(sub_directory)
                with open(f"{sub_directory}/{i}.txt", "w") as f:
                    f.write(str(torch.tensor([vocabulary[token] for token in self.tokenizer(reviews[i])], dtype=torch.long))+"\t"+str(data.loc[i, "label"]))

        # reuse the directories created earlier
        else:
            self.num_sub_directories = sum([os.path.isdir(directory_name+"/"+x) for x in os.listdir(directory_name)])
            self.num_examples_per_sub_directory = len(os.listdir(f"{directory_name}/0"))
            self.num_examples = self.num_examples_per_sub_directory*(self.num_sub_directories - 1) + len(os.listdir(f"{directory_name}/{self.num_sub_directories - 1}")) # to account for vocabulary file
            vocabulary = pickle.load(open(f"{vocabulary_location}/vocabulary.pickle", "rb"))
        
        print("Vocabulary size:", len(vocabulary.itos))
        global VOCABULARY_SIZE
        VOCABULARY_SIZE = len(vocabulary.itos)
        global PAD_INDEX
        PAD_INDEX = vocabulary['<pad>']
        global START_INDEX
        START_INDEX = vocabulary['<start>']
        global END_INDEX
        END_INDEX = vocabulary['<end>']
    
    # do some basic string formatting on the raw data
    def format(self, example):
        temp = re.sub('["#$%*+/<=>@\\^_`|~\t\n]', " ", str(example))
        add_space = {**{i:" "+i for i in '!&,:;'}, **{"\\"+i:" "+i for i in '.()[]{}?'}}
        for i in add_space.keys():
            temp = re.sub(r"{}".format(i), add_space[i], temp)
        return "<start> " + temp + " <end>"
    
    # pad zeros to make the equi dimensional tensors
    def padding_tensor(self, sequences, max_len, padding_value):
        num = len(sequences)
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
        for i, tensor in enumerate(sequences):
            length = min(tensor.size(0), max_len)
            out_tensor[i, :length] = tensor[:length]
        return out_tensor
    
    # create a batch of examples
    def generate_batch(self, data_batch):
        sequence_batch = []
        label_batch = []
        for s, l in data_batch:
            sequence_batch.append(s)
            label_batch.append(l)
        sequence_batch = self.padding_tensor(sequence_batch, max_len=MAX_LENGTH, padding_value=PAD_INDEX)
        sequence_batch = sequence_batch.permute(1,0)
        label_batch = torch.tensor(label_batch)
        return sequence_batch, label_batch

    # total number of available examples in the dataset    
    def __len__(self):
        return self.num_examples - self.num_examples%BATCH_SIZE
    
    # generate one example for a batch
    def __getitem__(self, idx):
        with open(f"{self.directory_name}/{idx//self.num_examples_per_sub_directory}/{idx}.txt", "r") as f:
            sequence, label = f.read().rsplit("\t", 1)
            sequence = eval("torch."+sequence)
            label = int(label)
            return sequence, label
