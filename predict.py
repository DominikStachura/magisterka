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
import pickle
import pandas as pd

from images_mean import generate_mean_images
from visualize_filters import show_img

# MODEL = Path('model_outputs/new/2/model_0_001_200_3.pt')
# MODEL = Path('model_outputs/new/granulacja/model_0_001_300_7.pt')
# MODEL = Path('model_outputs/new/parameters/model_0_001_200_5_Adam_xavier_False.pt')  # -> 74%
# MODEL = Path('model_outputs/new/with_labels/model_0_001_80_3_Adam_xavier_[False].pt')
# MODEL = Path('model_outputs/new/parameters/model_0_001_200_5_Adagrad_xavier_True.pt')
MODEL = Path('model_outputs/new/granulacja/model_0_001_200_5.pt')
num_classes = int(MODEL.stem.split('_')[-4])

# TO UNNORMALIZE IMG
invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                               transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                    std=[1., 1., 1.]),
                               ])

def convert_tensor_to_img(tensor):
    # tensor = tensor / 2 + 0.5
    return np.squeeze(np.transpose(tensor.cpu().numpy(), (1, 2, 0)))

def plot_class_histograms(model, images, labels, idx_to_class_mapping, num_classes=10, generate_output_images=False, output_images_name='output'):
    eng_mapping = {'kufel': 'mug', 'plyn': 'bottle', 'mis': 'teddy bear'}
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
        generate_mean_images(class_image_dict, f'mean_images_output/new/parameters/{output_images_name}_generated')
    plt.figure()
    for index, (label, prediction) in enumerate(predictions.items()):
        hist, bin_edges = np.histogram(prediction)
        bin_edges = np.round(bin_edges, 0)
        # plt.subplot(5, 2, index+1)
        plt.plot()
        plt.bar(bin_edges[:-1], hist, width=0.5)
        plt.xlim(-0.5, num_classes - 0.5)
        plt.xticks(np.arange(0, num_classes))
        plt.title(f'Label: {eng_mapping[idx_to_class_mapping[label]]}')
        plt.xlabel("Predicted class")
        plt.ylabel("Number of predicted classes")
        plt.show()


def compute_accuracy(model, preds_mapping=None, data_dir='datasets/new/', labels=None):
    # for all images
    transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    res = {}
    if labels:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataiter = iter(dataloader)
        images, labels = dataiter.next()
        correct = 0
        for img, label in zip(images, labels):
            output = torch.nn.Softmax()(model(img.unsqueeze(0).cuda()))
            _, pred = torch.max(output, 1)
            if pred == label:
                correct += 1
        res['total'] = 100 * correct / batch_size
        print(f"Total accuracy for labeled data: {res['total']:.2f}%")
        return res

    predictions = defaultdict(lambda: np.ndarray(0))
    predictions_neuron = defaultdict(lambda: np.ndarray(0))
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if '.jpg' in file:
                path = root + '/' + file
                img = Image.open(path)
                img = transform(img)
                output = torch.nn.Softmax()(model(img.unsqueeze(0).cuda()))
                # _, pred = torch.max(output, 1)
                # predictions[root.split('/')[-1]] = np.append(predictions[root.split('/')[-1]], preds_mapping[int(pred)])

                # 5 classess
                _, pred = torch.max(output, 1)
                for img_class, neurons in preds_mapping.items():
                    if int(pred) in neurons:
                        predicted_class = img_class
                        predicted_class_number = int(pred)
                        break
                predictions[root.split('/')[-1]] = np.append(predictions[root.split('/')[-1]], predicted_class)
                predictions_neuron[root.split('/')[-1]] = np.append(predictions_neuron[root.split('/')[-1]],
                                                                    predicted_class_number)

                # threshold set
                # above_threshold = False
                # for out in output.cpu().detach().numpy()[0]:
                #     if out > 0.8:
                #         above_threshold = True
                # if above_threshold:
                #     _, pred = torch.max(output, 1)
                #     predictions[root.split('/')[-1]] = np.append(predictions[root.split('/')[-1]], preds_mapping[int(pred)])
                # else:
                #     predictions[root.split('/')[-1]] = np.append(predictions[root.split('/')[-1]], preds_mapping[0])

        # more than 3 classess
    total = 0
    for img_class, neurons in preds_mapping.items():
        acc = 100 * (len(np.where(predictions[img_class] == img_class)[0]) / len(predictions[img_class]))
        print(f'{img_class} -> {acc:.2f}%')
        total += acc
        res[img_class] = f'{acc:.2f}%'
    print(f"Total accuracy: {total / len(preds_mapping.keys()):.2f}%")
    res['total'] = f'{total / len(preds_mapping.keys()):.2f}%'
    return res


def generate_models_comparison(data_dir, parameters):
    reports = []
    for param in parameters:
        model_path = param['model_path']
        model_params = torch.load(model_path)
        mapping = param.get('mapping', None)
        report = pd.DataFrame()
        report['Labels'] = [param.get('labels')]
        if 'optimizer' in model_params.keys():
            num_classes = model_path.stem.split('_')[-4]
            model = Net(num_classes)
            model.load_state_dict(model_params['model'])
            model.cuda()
            report['optimizer'] = [model_path.stem.split('_')[-3]]
            report['lr'] = [model_params['optimizer'].get('lr')]
            report['time_per_epoch'] = [model_params['time_per_epoch']]
        else:
            num_classes = model_path.stem.split('_')[-1]
            model = Net(num_classes)
            model.load_state_dict(model_params)
            model.cuda()
            report['optimizer'] = ['Adam']
            report['lr'] = [f'0.{model_path.stem.split("_")[-3]}']
            report['time_per_epoch'] = ['-']
        report['Num. of output classes'] = num_classes
        results = compute_accuracy(model, preds_mapping=mapping, data_dir=data_dir, labels=param.get('labels', False))
        report['bear acc.'] = [results.get('mis', '-')]
        report['pint acc.'] = [results.get('kufel', '-')]
        report['bottle acc.'] = [results.get('plyn', '-')]
        report['total acc.'] = [results.get('total', '-')]

        reports.append(report)

    pd.concat(reports).to_csv('report.csv', index=False)


class Net(nn.Module):
    def __init__(self, num_classes):
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
        self.fc3 = nn.Linear(128, int(num_classes))

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

model = Net(5)
model.cuda()

# ten maping jest dobierany na podstawie zdjec wyjsciowych z modelu
# preds_mapping = {
#     0: 'mis',
#     1: 'kufel',
#     2: 'plyn'
# }

preds_mapping = {
    'mis': [0],
    'kufel': [2, 3],
    'plyn': [1, 4]
}

preds_mapping = {
    'mis': [2],
    'kufel': [3, 4],
    'plyn': [0, 1]
}

model_params = torch.load(MODEL)
model.load_state_dict(model_params.get('model', model_params))

data_dir = 'datasets/new/'
transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
batch_size = len(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataiter = iter(dataloader)

images, labels = dataiter.next()

idx_to_class_mapping = {value: key for key, value in dataset.class_to_idx.items()}
compute_accuracy(model, preds_mapping, data_dir=data_dir, labels=False)
# plot_class_histograms(model, images, labels, idx_to_class_mapping, num_classes=5, generate_output_images=True)

# prepare data for reports
paramameters = []
paramameters.append({
    'model_path': Path('model_outputs/new/with_labels/model_0_001_80_3_Adam_xavier_[False].pt'),
    'labels': True,
})

paramameters.append({
    'model_path': Path('model_outputs/new/parameters/model_0_001_200_5_Adam_xavier_False.pt'),
    'labels': False,
    'mapping': {
        'mis': [3, 2],
        'kufel': [1, 4],
        'plyn': [0]
    }
})

paramameters.append({
    'model_path': Path('model_outputs/new/granulacja/model_0_001_300_7.pt'),
    'labels': False,
    'mapping': {
        'mis': [3],
        'kufel': [0, 1],
        'plyn': [2, 5, 6, 7]
    }
})

paramameters.append({
    'model_path': Path('model_outputs/new/2/model_0_001_200_3.pt'),
    'labels': False,
    'mapping': {
        'mis': [0],
        'kufel': [1],
        'plyn': [2]
    }
})

paramameters.append({
    'model_path': Path('model_outputs/new/parameters/model_0_001_200_3_Adadelta_xavier_True.pt'),
    'labels': False,
    'mapping': {
        'mis': [2],
        'kufel': [0],
        'plyn': [1]
    }
})

paramameters.append({
    'model_path': Path('model_outputs/new/parameters/model_0_001_200_5_Adagrad_xavier_True.pt'),
    'labels': False,
    'mapping': {
        'mis': [2,4],
        'kufel': [1],
        'plyn': [0, 3]
    }
})

# generate_models_comparison(data_dir, paramameters)

# for one image

# # # robi dobre predykcje
# img = Image.open('datasets/new/mis/augmented2.jpg')
# # img = Image.open('plyn_test.jpg')
# img = transform(img)
# output = model(img.unsqueeze(0).cuda())
# _, pred = torch.max(output, 1)
# print(preds_mapping[int(pred)])

# for all images
# predictions = defaultdict(lambda: np.ndarray(0))
# predictions_neuron = defaultdict(lambda: np.ndarray(0))
# for root, dirs, files in os.walk("datasets/new/"):
#     for file in files:
#         if '.jpg' in file:
#             path = root + '/' + file
#             img = Image.open(path)
#             img = transform(img)
#             output = torch.nn.Softmax()(model(img.unsqueeze(0).cuda()))
#             # _, pred = torch.max(output, 1)
#             # predictions[root.split('/')[-1]] = np.append(predictions[root.split('/')[-1]], preds_mapping[int(pred)])
#
#             # 5 classess
#             _, pred = torch.max(output, 1)
#             for img_class, neurons in preds_mapping.items():
#                 if int(pred) in neurons:
#                     predicted_class = img_class
#                     predicted_class_number = int(pred)
#                     break
#             predictions[root.split('/')[-1]] = np.append(predictions[root.split('/')[-1]], predicted_class)
#             predictions_neuron[root.split('/')[-1]] = np.append(predictions_neuron[root.split('/')[-1]],
#                                                                 predicted_class_number)
#
#             # threshold set
#             # above_threshold = False
#             # for out in output.cpu().detach().numpy()[0]:
#             #     if out > 0.8:
#             #         above_threshold = True
#             # if above_threshold:
#             #     _, pred = torch.max(output, 1)
#             #     predictions[root.split('/')[-1]] = np.append(predictions[root.split('/')[-1]], preds_mapping[int(pred)])
#             # else:
#             #     predictions[root.split('/')[-1]] = np.append(predictions[root.split('/')[-1]], preds_mapping[0])
#
# # more than 3 classess
# for img_class, neurons in preds_mapping.items():
#     acc = len(np.where(predictions[img_class] == img_class)[0]) / len(predictions[img_class])
#     print(f'{img_class} -> {acc}')

# check what neurons were activated by images from each class
# Counter(predictions_neuron['mis'])

# 3 classess
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
