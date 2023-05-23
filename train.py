import torch
import torch.optim
import time
from utils import *
from model import *
from model2 import *
import torch.backends.cudnn as cudnn

from sklearn.metrics import f1_score


# Model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None   # path to model checkpoint, None if none
batch_size = 128    # number of batches
# normally 4687
iterations = 4000
num_epochs = 15
learning_rate = 1e-2

cudnn.benchmark = True  # make learning faster depending on the GPU used


def main():
    model = CNN()
    # Move the model to the device
    model = model.to(device)
    # Defining the optimizer and loss function of the model
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_func = nn.CrossEntropyLoss()

    # Getting the data
    train_data, test_data = get_data()

    # Define the validation split ratio
    validation_split = 0.1
    
    # Compute the sizes of the validation and training sets
    dataset_size = len(train_data)
    validation_size = int(validation_split * dataset_size)
    train_size = dataset_size - validation_size

    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, validation_size])

    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=1),

        'validation' : torch.utils.data.DataLoader(val_data, 
                                            batch_size=batch_size, 
                                            num_workers=1),
        
        'test'  : torch.utils.data.DataLoader(test_data, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=1),
    }

    start = time.time()

    train(loaders=loaders, model=model, criterion=loss_func, optimizer=optimizer, num_epochs=num_epochs, validation=True)

    end = time.time()
    execution_time = end - start
    print(f"Execution time: {execution_time} seconds")

    save_checkpoint(num_epochs, model, optimizer)


def train(loaders, model, criterion, optimizer, num_epochs, validation = True):
    """
    Whole training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    for epoch in range(num_epochs):
        model.train()  # training mode enables dropout

        # used to calculate the loss across one entire epoch
        running_loss = 0.0
        correct = 0
        total = 0

        # One epoch
        total_step = len(loaders['train'])
        for i, (images, labels) in enumerate(loaders['train']):
            images, labels = images.to(device), labels.to(device)
            # images.shape => (batch_size, channels = 1, 28, 28)
            # labels.shape => (batch_size,)
            output = model(images)[0] #(N,10)
            # nn.CrossEntropyLoss() uses softmax
            loss = criterion(output, labels)
            # clear gradients for this training step   
            optimizer.zero_grad() 
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()

            # Track training loss and accuracy
            running_loss += loss.item()
            predicted = torch.max(output, 1)[1].data.squeeze()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, F1 Score: {:.4f}' .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), f1_score(labels.to('cpu'), predicted.to('cpu'), average='micro')))


            
        
        # Calculate average training loss and accuracy
        train_loss = running_loss / len(loaders['train'])
        train_accuracy = correct / total
            
        
        # display extensive data if true
        if validation:
            # Evaluate the model on the validation set
            model.eval()  # Set the model to evaluation mode
            validation_loss = 0.0
            validation_correct = 0
            validation_total = 0

            with torch.no_grad():
                for images, labels in loaders['validation']:
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    output = model(images)[0]
                    loss = criterion(output, labels)

                    # Track validation loss and accuracy
                    validation_loss += loss.item()
                    predicted = torch.max(output, 1)[1].data.squeeze()
                    validation_total += labels.size(0)
                    validation_correct += (predicted == labels).sum().item()

            # Calculate average validation loss and accuracy
            validation_loss = validation_loss / len(loaders['validation'])
            validation_accuracy = validation_correct / validation_total

            # Print the training and validation metrics for the epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {validation_loss:.4f}, Val Accuracy: {validation_accuracy:.4f}")




if __name__ == '__main__':
    main()