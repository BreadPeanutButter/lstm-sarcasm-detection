import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_loss(metrics):
    train, valid, epoch = metrics['train_loss_list'], metrics['valid_loss_list'], metrics['epoch_list']
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.title("Training & Validation Loss")
    plt.plot(epoch, train, 'b-o', label='Train')
    plt.plot(epoch, valid, 'g-o', label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png', bbox_inches='tight')

source_folder = '/home/j/joonjie/project'
plot_loss(metrics=torch.load(source_folder + '/metrics.pt'))
