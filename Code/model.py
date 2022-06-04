import string
import pandas as pd
import numpy as np
import torch
import torchtext
import time

from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn

# Cleans text by removing punctations and words with length <= 2 and making text lower case

def clean_text(text ): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>2))]) 
    return text2.lower()

# Converts topic names to numbers to work with the model

def topic_to_number(topic):
    if topic == 'Sports':
        return 0
    elif topic == 'Economics':
        return 1
    elif topic == 'Medicine':
        return 2
    
# Load the train and test datasets

train_data = pd.read_csv("../Training Data/train_data.csv")
test_data = pd.read_csv("../Testing Data/test_data.csv")

train_data['Topic'] = train_data['Topic'].apply(topic_to_number)
test_data['Topic'] = test_data['Topic'].apply(topic_to_number)

# Split the training data to train and validation

X_train, X_valid, Y_train, Y_valid= train_test_split(train_data['Article'].tolist(),\
                                                      train_data['Topic'].tolist(),\
                                                      test_size=0.2,\
                                                      stratify = train_data['Topic'].tolist(),\
                                                      random_state=57)
train_dat =list(zip(Y_train,X_train))
valid_dat =list(zip(Y_valid,X_valid))
test_dat=list(zip(test_data['Topic'].tolist(),test_data['Article'].tolist()))

print('Train data length:'+str(len(X_train)))
print('Topics distribution'+str(Counter(Y_train)))


print('Valid data length:'+str(len(X_valid)))
print('Topics distribution'+ str(Counter(Y_valid)))

print('Test data length:'+str(len(test_data['Article'].tolist())))
print('Topics distribution'+ str(Counter(test_data['Topic'].tolist())))
print('\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Work with the gpu (cuda)

tokenizer = get_tokenizer('basic_english') # torchtext tokenizer
train_iter = train_dat
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x)) # Build text pipleine using the vocab
label_pipeline = lambda x: int(x) 

# Collate function, needed for building pytorch dataloaders

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

# Building the model

class TopicClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TopicClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True) # A text embedding layer
        self.fc = nn.Linear(embed_dim, num_class) # A fully-connected layer
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    
num_class = 3
vocab_size = len(vocab)
emsize = 64 # Embedding size
model = TopicClassificationModel(vocab_size, emsize, num_class).to(device)

def train(dataloader):
    model.train()
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

BATCH_SIZE = 4
train_dataloader = DataLoader(train_dat, batch_size=BATCH_SIZE, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_dat, batch_size=BATCH_SIZE, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dat, batch_size=BATCH_SIZE, collate_fn=collate_batch)

# Hyperparameters

EPOCHS = 10 # epoch
LR = 2.8 # learning rate
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.2)
total_accu = None

# Start Training

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| End of Epoch: {:3d} | Time: {:5.2f}s | Validion Accuracy: {:8.3f} '.format(epoch, time.time() - epoch_start_time, accu_val))
    print('-' * 59)
    
accu_test = evaluate(test_dataloader)
print('Test Accuracy: {:8.3f}'.format(accu_test))

topic_label = {0:"Sports", 1: "Economics", 2: "Medicine"}

def predict(text, text_pipeline):
    with torch.no_grad():
        text = clean_text(text)
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return topic_label[output.argmax(1).item()]
    
ex_text_str = "Man City win the 2021/22 Premier League title: Six games that helped Pep Guardiola’s side become champions"
model = model.to("cpu")

print("\nTesting with text: Man City win the 2021/22 Premier League title: Six games that helped Pep Guardiola’s side become champions")
print("The Topic is: %s" %predict(ex_text_str, text_pipeline))