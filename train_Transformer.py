import torch
import random
import numpy as np

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# len(tokenizer.vocab)

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']       

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token


init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]                        # because of the start and the end
    return tokens




from torchtext import data
from torchtext import datasets

TEXT = data.Field(batch_first = True, use_vocab = False, tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids, init_token = init_token_idx,
                  eos_token = eos_token_idx, pad_token = pad_token_idx, unk_token = unk_token_idx)
LABEL = data.LabelField(dtype = torch.float)
# Set data format

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# train_data.examples = [{'text': [ids], 'label': 'pos' | 'neg'}, {...}, ...]
# train_data: dirname('aclImdb'), fields, name = ('imdb'), urls
print(vars(train_data.examples[6]))
# {'text': [1045, 2123, 1005, ... , 4276, 2009, 1012], 'label': 'pos'}
tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])
print(tokens)
# ['i', 'don', "'", ... , 'worth', 'it', '.']

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
# split training set & validation set

LABEL.build_vocab(train_data)
# make vocaburary with words in training data
print(LABEL.vocab.stoi)
# defaultdict(None, {'neg': 0, 'pos': 1})



##### BUILD MODEL #####

import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
import torch.optim as optim
import time

class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional,
                          batch_first = True, dropout = 0 if n_layers < 2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):    # text = [batch size, sent len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
        # embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        return output

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

bert = BertModel.from_pretrained('bert-base-uncased')
model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

optimizer = AdamW(model.parameters())
criterion = nn.BCEWithLogitsLoss()
# sigmoid layer + BCELoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins *60))
    return elapsed_mins, elapsed_secs


##### TRAIN #####
BATCH_SIZE = 64
N_EPOCHS = 5
best_valid_loss = float('inf')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                            batch_size = BATCH_SIZE, device = device)

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')