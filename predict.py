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

from visualize_filters import show_img

MODEL = Path('model_outputs/new/2/model_0_001_200_3.pt')
num_classes = int(MODEL.stem.split('_')[-1])

# TO UNNORMALIZE IMG
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

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
        x = self.maxpool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = Net()
model.cuda()



model.load_state_dict(torch.load(MODEL))
# robi dobre predykcje
img = Image.open('datasets/new/plyn/augmented1.jpg')
img = transform(img)
output = model(img.unsqueeze(0).cuda())
_, pred = torch.max(output, 1)
print(int(pred))

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