import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from copy import deepcopy

from collections import defaultdict



def convert_tensor_to_img(tensor):
    tensor = tensor.squeeze()
    # tensor = tensor / 2 + 0.5
    return np.transpose(tensor.cpu().numpy(), (1, 2, 0))


def show_img(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))


# loading data
data_dir = 'datasets/new/'
transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
batch_size = 300

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# # displaying images
# dataiter = iter(dataloader)
# images, labels = dataiter.next()
# images = images.numpy()  # convert images to numpy for display
#
# # plot the images in the batch, along with the corresponding labels
# fig = plt.figure(figsize=(25, 4))
# # display 20 images

# for idx in np.arange(len(images)):
#     ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
#     show_img(images[idx])
#     ax.set_title(classes[labels[idx]])
# plt.show()

# define architecture
class Net(nn.Module):
    def __init__(self, number_of_classes):
        super(Net, self).__init__()
        # poczatkowa architektura ktora stosowalem
        # self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        # self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        # self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        # self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        # self.maxpool = nn.MaxPool2d(2, 2)
        # # 2048 na wyjsciu z conv
        # self.fc1 = nn.Linear(4 * 4 * 128, 512)
        # self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(128, num_classes)

        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv5 = nn.Conv2d(64, 64, 5, padding=2)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4 * 4 * 64, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, number_of_classes)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.maxpool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# implement xavier
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)



def find_n_best(output, n=1, num_classes=6):
    # {target0: [zdj1, zdj2...], target1:[...]}
    dataset = defaultdict(list)
    output = output.cpu().detach().numpy()
    output_to_update = deepcopy(output)
    for i in range(num_classes):
        best = sorted(output_to_update, key=lambda x: x[i])[::-1][:n]
        for el in best:
            dataset[i].append(np.where(output == el)[0][0])
            output_to_update = np.delete(output_to_update, np.where(output_to_update == el)[0][0], axis=0)
        # for j in range(n):
        #     # data_index = np.where(output == max(output, key=lambda x: x[i]))[0][0]
        #     sorted_outputs = sorted(output, key=lambda x: x[i])[::-1]
        #     best = np.where(output == sorted_outputs[0])
        #     for el in best:
        #
        #     dataset[i].append(data_index)
            # usuwam po dodaniu, bo na poczatku nie wiem czemu zawsze jeden ma wszystkie maxy, i to by znowu zepsulo caly proces
            # output = np.delete(output, data_index, axis=0)
    return dataset


# check for different parameters

lrs = [0.001]
epochs = [80, 200]
classes = [3]
xavier = [False]
opts = [optim.Adam]#, optim.Adadelta, optim.Adagrad]

for opt in opts:
    for lr in lrs:
        for num_epoch in epochs:
            for num_classes in classes:

                model = Net(num_classes)
                if xavier:
                    model.apply(weights_init)
                # model.cuda()

                criterion = nn.CrossEntropyLoss()
                # optimizer = optim.Adam(model.parameters(), lr=lr)
                optimizer = opt(model.parameters())

                start_time = time.time()
                for epoch in range(1, num_epoch):
                    # najpierw sprawdzam co wypluje model i na tej podstawie dobieram labele
                    # model.eval()
                    # wewnetrzne fory wykonuja sieraz bo batch size wielkosci calego zbioru
                    for data, labels in dataloader:
                        # data = data.cuda()
                        # labels = labels.cuda()

                        optimizer.zero_grad()
                        output = model(data)

                        loss = criterion(output, labels)

                        loss.backward()

                        optimizer.step()

                    if epoch % 5 == 0:
                        print(f'Epoch: {epoch}')

                end_time = time.time()
                time_for_epoch = (end_time-start_time) / num_epoch
                torch.save({'model': model.state_dict(),
                            'time_per_epoch': time_for_epoch,
                            'optimizer': optimizer.defaults},
                            f'model_outputs/new/parameters/model_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}_{type(optimizer).__name__}_xavier_{str(xavier)}.pt')



