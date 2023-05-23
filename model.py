import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Simple CNN architecture for the MNIST problem
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layer, output 10 classes
        self.lin1 = nn.Linear(in_features=7*7*64, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=10)


    def forward(self, image):
        """
        Forward propagation.

        :param image: image, a tensor of dimensions (to determine)
        :return: to be determined
        """
        out = F.relu(self.conv1(image))
        out = self.pool1(out)

        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        out = out.view(out.size(0), -1)
        out = F.dropout(self.lin1(out),p=0.5)
        output = self.lin2(out)



        # x = self.conv1(x)
        # x = self.conv2(x)
        # # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # x = x.view(x.size(0), -1)       
        # output = self.out(x)
        return output, out    # return x for visualization