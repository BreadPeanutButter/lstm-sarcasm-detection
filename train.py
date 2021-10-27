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

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, classification_report, confusion_matrix

import numpy as np

import datetime

import matplotlib.pyplot as plt
import seaborn as sns

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
        
        out = self.out(hidden1)
        out = torch.squeeze(out, 1)

        text_out = torch.sigmoid(out)

        return text_out

def train(model, optimizer, scheduler, train_loader,
          valid_loader,valid_every, # validate every how many batches
          criterion = nn.BCELoss(), num_epochs = 1):
  
    start = datetime.datetime.now()
    # initialize running values
    running_loss = 0.0
    local_batch_num = 0
    global_batch_num = 0
    valid_running_loss = 0.0
    valid_loss_list = []
    valid_accuracy_list = []
    train_loss_list = []
    epoch_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        local_batch_num = 0
        running_loss = 0

        # Validate once at the beginning
        if global_batch_num == 0:
            y_pred = []
            y_true = []
            model.eval()
            with torch.no_grad():                    
            # validation loss loop
                for (labels, (comment, comment_len)), _ in valid_loader:
                    labels = labels.to(device)
                    comment = comment.to(device)
                    comment_len = comment_len.to(device)
                    output = model(comment, comment_len)

                    prediction = (output > 0.5).int()
                    y_pred.extend(prediction.tolist())
                    y_true.extend(labels.tolist())

                    loss = criterion(output, labels)
                    valid_running_loss += loss.item()       

            # validation
            average_valid_loss = valid_running_loss / len(valid_loader)
            valid_accuracy = accuracy_score(y_true, y_pred)
            valid_loss_list.append(average_valid_loss)
            valid_accuracy_list.append(valid_accuracy)
            epoch_list.append(0)
            valid_running_loss = 0.0     
            print('Epoch [{}/{}], Batch [{}/{}], Valid Loss: {:.4f}, Valid Accuracy: {:.4f}'
                    .format(epoch, num_epochs, local_batch_num, len(train_loader),
                            average_valid_loss, valid_accuracy))
        model.train()

        for (labels, (comment, comment_len)), _ in train_loader:
            labels = labels.to(device)
            comment = comment.to(device)
            comment_len = comment_len.to(device)
            output = model(comment, comment_len)
            loss = criterion(output, labels)

            if global_batch_num == 0:
                train_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            local_batch_num += 1
            global_batch_num += 1

            # validation step
            if local_batch_num % valid_every == 0:
                y_pred = []
                y_true = []
                model.eval()
                with torch.no_grad():                    
                # validation loss loop
                    for (labels, (comment, comment_len)), _ in valid_loader:
                        labels = labels.to(device)
                        comment = comment.to(device)
                        comment_len = comment_len.to(device)
                        output = model(comment, comment_len)

                        prediction = (output > 0.5).int()
                        y_pred.extend(prediction.tolist())
                        y_true.extend(labels.tolist())

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()       

                # validation
                average_train_loss = running_loss / valid_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                valid_accuracy = accuracy_score(y_true, y_pred)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                valid_accuracy_list.append(valid_accuracy)
                epoch_list.append(global_batch_num/len(train_loader))

                # resetting running values
                running_loss = 0.0    
                valid_running_loss = 0.0     
                model.train()

                # print progress
                print('Epoch [{}/{}], Batch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}'
                .format(epoch+1, num_epochs, local_batch_num, len(train_loader),
                average_train_loss, average_valid_loss, valid_accuracy))
        scheduler.step()

    end = datetime.datetime.now()
    torch.save(model.state_dict(), source_folder + '/lstm_model.pt')

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'valid_accuracy_list': valid_accuracy_list,
                  'epoch_list': epoch_list}
    torch.save(state_dict, source_folder + '/metrics.pt')

    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))
    print('Model weights and metrics saved.')

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

def plot_loss(metrics):
    train, valid, epoch = metrics['train_loss_list'], metrics['valid_loss_list'], metrics['epoch_list']
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.title("Loss against Epoch")
    plt.plot(epoch, train, '.b-', label='Training')
    plt.plot(epoch, valid, '.g-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png', bbox_inches='tight')
    plt.cla()
    plt.clf()

def plot_accuarcy(metrics):
    accuracy, epoch = metrics['valid_accuracy_list'], metrics['epoch_list']
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.title("Validation Accuracy against Epoch")
    plt.plot(epoch, accuracy, '.m-', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.savefig('accuracy.png', bbox_inches='tight')


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
    valid_iter = BucketIterator(valid_split, batch_size=512, sort_key=lambda x: len(x.comment),
                            device=device, sort=True, sort_within_batch=True)

    # Vocabulary
    text_field.build_vocab(train_split, min_freq=3)

    # Train
    model = LSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.80)
    train(model=model, optimizer=optimizer, scheduler=scheduler, 
    train_loader=train_iter, valid_loader=valid_iter, num_epochs=5, valid_every=len(train_iter)//10) # Evaluate 10 times per epoch

    # Validation
    trained_model = LSTM().to(device)
    trained_model.load_state_dict(torch.load(source_folder + '/lstm_model.pt'))
    validate(trained_model, valid_iter)
    metrics = torch.load(source_folder + '/metrics.pt')
    plot_loss(metrics)
    plot_accuarcy(metrics)
