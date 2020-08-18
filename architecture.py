import torch.nn as nn
import torch.nn.functional as F


# define architecture
class Net(nn.Module):

    def __init__(self, num_classes, mnist=False):
        """
        Architecture of neural network
        :param mnist: indicates if the architecture will be used for MNIST dataset
        """
        super(Net, self).__init__()
        # for mnist
        if mnist:
            self.conv1 = nn.Conv2d(1, 128, 3, padding=1)  # -> 14 after pooling
            self.conv2 = nn.Conv2d(128, 128, 3, padding=2)  # -> 8 after pooling
            self.conv3 = nn.Conv2d(128, 64, 3, padding=1)  # -> 4 after pooling
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

            self.maxpool = nn.MaxPool2d(2, 2)

            self.fc1 = nn.Linear(4 * 4 * 64, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, num_classes)

        else:
            self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
            self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
            self.conv5 = nn.Conv2d(64, 64, 5, padding=2)
            self.maxpool = nn.MaxPool2d(2, 2)

            self.fc1 = nn.Linear(4 * 4 * 64, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
