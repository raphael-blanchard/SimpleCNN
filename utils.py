import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data():
    """
    Return the MNIST datasets

    :return: train, test datasets
    """
    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )
    return train_data, test_data

def save_checkpoint(epoch, model, optimizer):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_CNN_MNIST.pth.tar'   # .pth.tar file to keep in memory the optimizer and epochs
    torch.save(state, filename)