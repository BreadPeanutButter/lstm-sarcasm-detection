import os
# Libraries
import torch

# Preliminaries
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

# Models
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ExponentialLR

# Training
import torch.optim as optim

from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

import numpy as np

import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.manual_seed(0)

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
        #self.fc2 = nn.Linear(2*dimension, dimension)
        #self.fc3 = nn.Linear(dimension, dimension)
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

        hidden1 = self.fc1(encoding)
        hidden1 = self.relu(hidden1)
        hidden1 = self.drop(hidden1)

        #hidden2 = self.fc2(hidden1)
        #hidden2 = self.relu(hidden2)
        #hidden2 = self.drop(hidden2)

        #hidden3 = self.fc3(hidden2)
        #hidden3 = self.relu(hidden3)
        #hidden3 = self.drop(hidden3)
        
        out = self.out(hidden1)
        out = torch.squeeze(out, 1)

        text_out = torch.sigmoid(out)

        return text_out

def train(model,
          optimizer,
          scheduler,
          train_loader,
          eval_every,
          criterion = nn.BCELoss(),
          num_epochs = 20):
  
    start = datetime.datetime.now()
    # initialize running values
    running_loss = 0.0
    global_step = 0
    train_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, (comment, comment_len)), _ in train_loader:
            try:
              labels2 = labels.to(device)
              comment = comment.to(device)
              comment_len = comment_len.to(device)
              output = model(comment, comment_len)

              loss = criterion(output, labels2)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
            except:
              print(labels)
              raise 

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()       

                # evaluation
                average_train_loss = running_loss / eval_every
                train_loss_list.append(average_train_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss))
        scheduler.step()

    end = datetime.datetime.now()
    torch.save(model.state_dict(), source_folder + '/lstm_model.pt')

    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))
    print('Model weights saved.')

def validate(model, valid_loader):
    y_pred = []
    y_true = []
    y_pred_raw = []

    model.eval()
    with torch.no_grad():
        for (labels, (comment, comment_len)), _ in valid_loader:           
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
    print('Confusion Matrix')
    print(cm)

    print('ROC')
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_raw)
    optimal_threshold = thresholds[np.argmax(tpr-fpr)]
    auc = roc_auc_score(y_true, y_pred_raw)
    print('auc: ', auc)
    print('Optimal threshold: ', optimal_threshold)

    y_pred_optimal = [int(x > optimal_threshold) for x in y_pred_raw]
    print('Optimal Classification Report:')
    print(classification_report(y_true, y_pred_optimal, labels=[1,0], target_names=['Positive', 'Negative'], digits=4))
    cm2 = confusion_matrix(y_true, y_pred_optimal, labels=[1,0])
    print('Optimal Confusion Matrix')
    print(cm2)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
    fields = [('label', label_field), ('comment', text_field )]

    source_folder = '/home/j/joonjie/project'
    # TabularDataset
    train_split, valid_split = TabularDataset.splits(path=source_folder, train='sarcasm_train.csv', validation='sarcasm_valid.csv',
                                           format='CSV', fields=fields, skip_header=True)

    # Iterators
    train_iter = BucketIterator(train_split, batch_size=512,shuffle=True, sort_key=lambda x: len(x.comment),
                            device=device, sort=None, sort_within_batch=None)
    valid_iter = BucketIterator(valid_split, batch_size=256, sort_key=lambda x: len(x.comment),
                            device=device, sort=True, sort_within_batch=True)

    # Vocabulary
    text_field.build_vocab(train_split, min_freq=3)

    # Train
    model = LSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.80)
    train(model=model, optimizer=optimizer, scheduler=scheduler, 
    train_loader=train_iter, num_epochs=2, eval_every=len(train_iter)//10)

    # Validation
    trained_model = LSTM().to(device)
    trained_model.load_state_dict(torch.load(source_folder + '/lstm_model.pt'))
    validate(trained_model, valid_iter)
