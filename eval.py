import torch
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = "E:\CompVision\MNIST_CNN\checkpoint_CNN_MNIST.pth.tar"
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

train_data, test_data = get_data()

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=1) 
nb_train = len(train_data)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=1) 
nb_test = len(test_data)

epoch = checkpoint['epoch']

def evaluate(test_loader, model):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            pass
    
    print("Loaded model from epoch:", epoch)
    print('Test Accuracy of the model on the 10000 test images: %.4f' % (correct/len(test_loader.dataset)))
    print("Nb of corrects: ", correct)
    
    pass
    return images, labels, pred_y

import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(images, labels, predictions, num_samples):
    # Select a random subset of samples
    print(images.shape)
    indices = np.random.choice(len(images), num_samples, replace=False)
    images = images[indices]
    labels = labels[indices]
    predictions = predictions[indices]

    fig, axes = plt.subplots(1, num_samples, figsize=(12, 2))
    fig.suptitle('MNIST Predictions', fontsize=16)

    for i, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {label}\nPred: {prediction}')
        axes[i].axis('off')

    plt.show()

# Assuming 'images' is the evaluation images, 'labels' are the true labels,
# and 'predictions' are the predicted labels


if __name__ == '__main__':
    img, lab, pred = evaluate(test_loader, model)
    img, lab, pred = img.to('cpu'), lab.to('cpu'), pred.to('cpu')
    img = img.squeeze(axis = 1)
    visualize_predictions(img, lab, pred, num_samples=16)