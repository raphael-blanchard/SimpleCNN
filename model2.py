import torch.nn as nn
import torch.nn.functional as F


class CNN2(nn.Module):
    """
    Simple CNN architecture for the MNIST problem
    """

    def __init__(self):
        super(CNN2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(2)

        # fully connected layer, output 10 classes
        self.out = nn.Linear(in_features=32*7*7, out_features=10)


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
        output = self.out(out)


        # x = self.conv1(x)
        # x = self.conv2(x)
        # # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # x = x.view(x.size(0), -1)       
        # output = self.out(x)
        return output, out    # return x for visualization