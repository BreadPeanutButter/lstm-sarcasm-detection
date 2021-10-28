import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_loss(metrics):
    train, valid, epoch = metrics['train_loss_list'], metrics['valid_loss_list'], metrics['epoch_list']
    plt.cla()
    plt.clf()
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.title("Loss against Epoch")
    plt.plot(epoch, train, '.b-', label='Training')
    plt.plot(epoch, valid, '.g-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('valid_loss.png', bbox_inches='tight')
    plt.cla()
    plt.clf()

def plot_accuarcy(metrics):
    accuracy, epoch = metrics['valid_accuracy_list'], metrics['epoch_list']
    plt.cla()
    plt.clf()
    sns.set(style='darkgrid')
    plt.rcParams["figure.figsize"] = (12,6)
    plt.title("Validation Accuracy against Epoch")
    plt.plot(epoch, accuracy, '.m-', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('valid_accuracy.png', bbox_inches='tight')

source_folder = '/home/j/joonjie/project'
metrics = torch.load(source_folder + '/metrics.pt')
plot_loss(metrics)
plot_accuarcy(metrics)
