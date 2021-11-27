import os

# Libraries
import torch
import numpy as np

# Preliminaries
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

# Models
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Training
import torch.optim as optim

# Evaluation
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

import csv

torch.manual_seed(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class LSTM(nn.Module):

    def __init__(self, dimension=512):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2*dimension, 2*dimension)
        self.out = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        encoding_forward = output[range(len(output)), text_len - 1, :self.dimension]
        encoding_reverse = output[:, 0, self.dimension:]
        encoding_reduced = torch.cat((encoding_forward, encoding_reverse), 1)
        encoding = self.drop(encoding_reduced)

        hidden = self.fc1(encoding)
        hidden = self.relu(hidden)
        hidden = self.drop(hidden)

        out = self.out(hidden)
        out = torch.squeeze(out, 1)

        text_out = torch.sigmoid(out)

        return text_out

# Evaluation Function

def evaluate(model, test_loader, threshold=0.5):
    y_pred = []
    y_true = []
    y_pred_raw = []

    model.eval()
    with torch.no_grad():
        for (labels, (comment, comment_len)), _ in test_loader:           
            labels = labels.to(device)
            comment = comment.to(device)
            comment_len = comment_len.to(device)
            output = model(comment, comment_len)
            prediction = (output > 0.5).int()
            y_pred_raw.extend(output.tolist())
            y_pred.extend(prediction.tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], target_names=['Positive', 'Negative'], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    print(cm)

    print('ROC')
    auc = roc_auc_score(y_true, y_pred_raw)
    print('auc: ', auc)
    
    print(len(y_pred))
    with open('lstm_out.csv', 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        for i in y_pred:
            wr.writerow([i])


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
    fields = [('label', label_field), ('comment', text_field )]

    source_folder = '/home/j/joonjie/project'
    # TabularDataset
    train_split, test_split = TabularDataset.splits(path=source_folder, train='sarcasm_train_valid.csv', test='sarcasm_test.csv',
                                           format='CSV', fields=fields, skip_header=True)

    # Iterators
    test_iter = BucketIterator(test_split, batch_size=512, shuffle=False,
                            device=device, sort=False)

    # Vocabulary
    text_field.build_vocab(train_split, min_freq=3)

    # Evaluate
    trained_model = LSTM().to(device)
    trained_model.load_state_dict(torch.load(source_folder + '/lstm_model_final.pt'))
    evaluate(trained_model, test_iter)
