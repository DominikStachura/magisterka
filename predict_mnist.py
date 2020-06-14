import torch
from PIL import Image
from torchvision import datasets, transforms
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import defaultdict
import os
from collections import Counter
from torchvision import datasets

from visualize_filters import show_img

# MODEL = Path('model_outputs/new/2/model_0_001_200_3.pt')
MODEL = Path('model_outputs/new/mnist/model_0_001_50_10.pt')
num_classes = int(MODEL.stem.split('_')[-1])


transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataset = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)

batch_size = 5000
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_classes = 10
classes = dataset.classes


# # displaying images
dataiter = iter(dataloader)
images, labels = dataiter.next()
# images = images.numpy()  # convert images to numpy for display

class Net(nn.Module):
    def __init__(self):
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

        # for mnist
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)  # -> 14 after pooling
        self.conv2 = nn.Conv2d(128, 128, 3, padding=2)  # -> 8 after pooling
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)  # -> 4 after pooling
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

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



model = Net()
model.cuda()

# ten maping jest dobierany na podstawie zdjec wyjsciowych z modelu
preds_mapping = {
    6: [0],
    8: [1],
    7: [2],
    3: [3, 7],
    1: [4],
    5: [5],
    0: [6],
    9: [8, 9]
}

model.load_state_dict(torch.load(MODEL))


# for one image

# # # robi dobre predykcje
# img = Image.open('datasets/new/mis/augmented2.jpg')
# # img = Image.open('plyn_test.jpg')
# img = transform(img)
# output = model(img.unsqueeze(0).cuda())
# _, pred = torch.max(output, 1)
# print(preds_mapping[int(pred)])

# for all images
predictions = defaultdict(lambda: np.ndarray(0))
predictions_neuron = defaultdict(lambda: np.ndarray(0))

for img, label in zip(images, labels):
    output = torch.nn.Softmax()(model(img.unsqueeze(0).cuda()))
    _, pred = torch.max(output, 1)
    predicted_class = None
    for img_class, neurons in preds_mapping.items():
        if int(pred) in neurons:
            predicted_class = img_class
            break
    predictions[int(label)] = np.append(predictions[int(label)], predicted_class)
    predictions_neuron[int(label)] = np.append(predictions[int(label)], int(pred))


# more than 3 classess
for img_class, neurons in preds_mapping.items():
    acc = len(np.where(predictions[img_class]==img_class)[0])/len(predictions[img_class])
    print(f'{img_class} -> {acc:.2} % accuracy')

# check what neurons were activated by images from each class
# Counter(predictions_neuron['mis'])

#3 classess
# for img_class in preds_mapping.values():
#     acc = len(np.where(predictions[img_class]==img_class)[0])/len(predictions[img_class])
#     print(f'{img_class} -> {acc}')

# preds = defaultdict(int)
#
# # imgs = glob.glob('datasets/kufel/*.png')
# imgs = glob.glob('datasets/new/*.jpg')
# for img in imgs:
#     img = Image.open(img)
#     img = transform(img)
#     # plt.imshow(np.transpose(invTrans(img), (1, 2, 0)))
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     plt.show()
#     # output = model(img.cuda())
#     # _, pred = torch.max(output, 1)
#     # preds[int(pred)] += 1
#     # print(int(pred))
#
# for cls, val in preds.items():
#     print(cls, '  -->  ', val)