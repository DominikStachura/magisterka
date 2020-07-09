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
import pickle
from images_mean import generate_mean_images

from visualize_filters import show_img

# MODEL = Path('model_outputs/new/2/model_0_001_200_3.pt')
MODEL = Path('model_outputs/new/mnist/model_0_001_80_10.pt')
num_classes = int(MODEL.stem.split('_')[-1])


transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.MNIST(root='data', train=True,
                         download=True, transform=transform)



# # displaying images
# dataiter = iter(dataloader)
# images, labels = dataiter.next()
# images, labels = dataiter.next()


# images = images.numpy()  # convert images to numpy for display
def convert_tensor_to_img(tensor):
    # tensor = tensor / 2 + 0.5
    return np.squeeze(np.transpose(tensor.cpu().numpy(), (1, 2, 0)))

def plot_class_histograms(model, images, labels, num_classes = 10, generate_output_images=True, output_images_name='output'):
    class_image_dict = defaultdict(list)
    predictions = defaultdict(list)
    for label, img in zip(labels, images):
        output = torch.nn.Softmax()(model(img.unsqueeze(0).cuda()))
        _, pred = torch.max(output, 1)
        predictions[int(label)].append(int(pred))
        if generate_output_images:
            class_image_dict[int(pred)].append(convert_tensor_to_img(img))

    if generate_output_images:
        # with open(f'pickle_outputs/new/mnist/class_image_dict_manually_generated.pickle', 'wb') as f:
        #     pickle.dump(class_image_dict, f)
        generate_mean_images(class_image_dict, f'mean_images_output/new/mnist/{output_images_name}_generated')

    plt.figure()
    for index, (label, prediction) in enumerate(predictions.items()):
        hist, bin_edges = np.histogram(prediction)
        bin_edges = np.round(bin_edges, 0)
        # plt.subplot(5, 2, index+1)
        plt.plot()
        plt.bar(bin_edges[:-1], hist, width=0.5)
        plt.xlim(-0.5, num_classes-0.5)
        plt.xticks(np.arange(0, num_classes))
        plt.title(f'Label: {label}')
        plt.xlabel("Predicted class")
        plt.ylabel("Number of predicted classes")
        plt.show()


def compute_accuracy(model, images, labels, preds_mapping):
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
        acc = len(np.where(predictions[img_class] == img_class)[0]) / len(predictions[img_class])
        print(f'{img_class} -> {acc:.2} % accuracy')


def create_class_imgs_dict(model, images):
    class_image_dict = defaultdict(list)
    for img in images:
        output = torch.nn.Softmax()(model(img.unsqueeze(0).cuda()))
        _, pred = torch.max(output, 1)
        class_image_dict[int(pred)].append(convert_tensor_to_img(img))

    with open(f'pickle_outputs/new/mnist/class_image_dict_manually_generated.pickle', 'wb') as f:
        pickle.dump(class_image_dict, f)




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
    1: [2],
    6: [1],
    0: [6],
    8: [4],
    9: [0, 3]
}

model.load_state_dict(torch.load(MODEL))


batch_size = 30000
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

classes = dataset.classes
dataiter = iter(dataloader)
images, labels = dataiter.next()

# histograms

plot_class_histograms(model, images, labels, num_classes=num_classes, output_images_name=MODEL.stem)

# create_class_imgs_dict(model, images)
# compute_accuracy(model, images, labels, preds_mapping)

