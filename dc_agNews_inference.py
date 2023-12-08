import torchtext
print(torchtext.__version__)

import site
print(site.getsitepackages())
import os
os.environ['SP_DIR'] = '/opt/conda/lib/python3.11/site-packages'

import torch
import pandas as pd
import re

# nlp library of Pytorch
from torchtext import data

import warnings as wrn
wrn.filterwarnings('ignore')

import torch.nn.functional as F

SEED = 2021

torch.manual_seed(SEED)
torch.backends.cuda.deterministic = True

df = pd.read_csv('./ag_news_csv/train.csv', header=None)

def clean_text(text):
    # Replace any non-alphanumeric character with a space
    cleaned_text = re.sub(r'[^A-Za-z0-9]+', ' ', str(text))
    return cleaned_text

df[0] = df[0].apply(clean_text)
df[1] = df[1].apply(clean_text)
df[2] = df[2].apply(clean_text)

df.to_csv('./ag_news_csv/cleaned_train.csv', index=False, header=False)

data_=pd.read_csv('./ag_news_csv/cleaned_train.csv', header=None)
data_.head()

data_.info()

# Field is a normal column 
# LabelField is the label column.
#from torchtext.legacy import data

# TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
# LABEL = data.LabelField(dtype = torch.float,batch_first=True)

import spacy
from torchtext.data.utils import get_tokenizer

# Make sure to load the correct spacy model
spacy_en = spacy.load('en_core_web_sm')

# Define the tokenizer function
def spacy_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Assuming you're using the legacy torchtext API
#from torchtext.legacy import data
#TEXT = data.Field(tokenize=spacy_tokenizer, batch_first=True, include_lengths=True)
#LABEL = data.LabelField(dtype=torch.float, batch_first=True)

LABEL = data.LabelField()
TEXT = data.Field(tokenize=spacy_tokenizer, batch_first=True, include_lengths=True)
fields = [("label", LABEL), ("text", TEXT)]

#fields = [("type",LABEL),('text',TEXT)]

training_data = data.TabularDataset(path="./ag_news_csv/cleaned_train.csv",
                                    format="csv",
                                    fields=fields,
                                    skip_header=True
                                   )

print(vars(training_data.examples[0]))

test_data = data.TabularDataset(
            path="./ag_news_csv/test.csv", 
            format="csv",
            fields=fields,
            skip_header=True)

import random
# train and validation splitting
train_data,valid_data = training_data.split(split_ratio=0.75,
                                            random_state=random.seed(SEED))

# Building vocabularies => (Token to integer)
TEXT.build_vocab(train_data,
                 min_freq=5)

LABEL.build_vocab(train_data)
# Count the number of instances per class
label_counts = {LABEL.vocab.itos[i]: LABEL.vocab.freqs[LABEL.vocab.itos[i]] for i in range(len(LABEL.vocab))}
print("Number of instances per class:", label_counts)


print("Size of text vocab:",len(TEXT.vocab))

print("Size of label vocab:",len(LABEL.vocab))

TEXT.vocab.freqs.most_common(10)

# Creating GPU variable
device = torch.device("cuda")

BATCH_SIZE = 32

# We'll create iterators to get batches of data when we want to use them
"""
This BucketIterator batches the similar length of samples and reduces the need of 
padding tokens. This makes our future model more stable

"""
train_iterator,validation_iterator = data.BucketIterator.splits(
    (train_data,valid_data),
    batch_size = BATCH_SIZE,
    # Sort key is how to sort the samples
    sort_key = lambda x:len(x.text),
    sort_within_batch = True,
    device = device
)

test_iterator = data.BucketIterator(
            test_data, 
            batch_size=BATCH_SIZE, 
            sort_key=lambda x: len(x.text),
            sort_within_batch=True, 
            device=device)

# Pytorch's nn module has lots of useful feature
import torch.nn as nn

class LSTMNet(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout):
        
        super(LSTMNet,self).__init__()
        
        # Embedding layer converts integer sequences to vector sequences
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        
        # LSTM layer process the vector sequences 
        # self.lstm = nn.LSTM(embedding_dim,
        #                     hidden_dim,
        #                     num_layers = n_layers,
        #                     bidirectional = bidirectional,
        #                     dropout = dropout,
        #                     batch_first = True
        #                    )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Adjusted from hidden_dim * 2 to hidden_dim
        # Dense layer to predict 
        #self.fc = nn.Linear(hidden_dim * 2,output_dim)
        # Prediction activation function
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)
        
        # Thanks to packing, LSTM don't see padding tokens 
        # and this makes our model better
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        
        
        packed_output,(hidden_state,cell_state) = self.lstm(packed_embedded)
        # Concatenating the final forward and backward hidden states
        #hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        hidden = hidden_state[-1,:,:]
        
        dense_outputs=self.fc(hidden)

        output = F.softmax(dense_outputs, dim=1)

        log_output = torch.log(output)

        #Final activation function
        #outputs=self.sigmoid(dense_outputs)
        
        #return outputs
        return log_output

SIZE_OF_VOCAB = len(TEXT.vocab)
EMBEDDING_DIM = 100
NUM_HIDDEN_NODES = 100
NUM_OUTPUT_NODES = len(LABEL.vocab)
NUM_LAYERS = 1
BIDIRECTION = False
DROPOUT = 0

model = LSTMNet(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT)
#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()

#!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

import torch
print(torch.cuda.is_available())

import torch.optim as optim
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
#criterion = nn.BCELoss()
#criterion = criterion.to(device)

model

# We'll use this helper to compute accuracy
# def binary_accuracy(preds, y):
#     #round predictions to the closest integer
#     rounded_preds = torch.round(preds)
    
#     correct = (rounded_preds == y).float() 
#     acc = correct.sum() / len(correct)
#     return acc

def multi_class_accuracy(preds, y):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model,iterator,optimizer,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    model.train()
    
    for batch in iterator:
        
        # cleaning the cache of optimizer
        optimizer.zero_grad()
        
        text,text_lengths = batch.text
        
        # forward propagation and squeezing
        predictions = model(text,text_lengths).squeeze()
        
        # computing loss / backward propagation
        loss = criterion(predictions, batch.label)
        #loss = criterion(predictions,batch.type)
        loss.backward()
        
        # accuracy
        acc = multi_class_accuracy(predictions,batch.label)
        
        # updating params
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    # It'll return the means of loss and accuracy
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model,iterator,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    # deactivate the dropouts
    model.eval()
    
    # Sets require_grad flat False
    with torch.no_grad():
        for batch in iterator:
            text,text_lengths = batch.text
            
            predictions = model(text,text_lengths).squeeze()
              
            #compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = multi_class_accuracy(predictions, batch.label)
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# EPOCH_NUMBER = 100
# print("total number of epoch:",EPOCH_NUMBER)
# for epoch in range(1,EPOCH_NUMBER+1):
    
#     print("======================================================")
#     print("Epoch: %d" %epoch)
#     print("======================================================")
#     train_loss,train_acc = train(model,train_iterator,optimizer,criterion)
    
#     valid_loss,valid_acc = evaluate(model,validation_iterator,criterion)
    
#     # Showing statistics
#     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
#     print()

# model_parameters = model.state_dict()
# print(model_parameters)
# torch.save(model.state_dict(), './modelParameter2.pth')

model.load_state_dict(torch.load('./modelParameter2.pth'))

# Evaluate the model on test data
test_loss, test_acc = evaluate(model, test_iterator, criterion)

# Print the accuracy
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

