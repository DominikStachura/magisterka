import torch
from PIL import Image
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from collections import defaultdict

from images_mean import generate_mean_images
from architecture import Net

# defining path for the model we want to validate
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


def plot_class_histograms(model, images, labels, idx_to_class_mapping, num_classes=10, generate_output_images=False,
                          output_images_name='output'):
    """
    Mfunction used for generating histograms of samples assigned to each class
    :param model: trained model
    :param images: images for predictions
    :param labels: labels for given images
    :param idx_to_class_mapping: predicted output class mapping
    :param num_classes: number of output classes
    :param generate_output_images: boolean value to indicate if we want to generate mean images
    :param output_images_name: name for figure generate from mean images
    """
    eng_mapping = {'kufel': 'mug', 'plyn': 'bottle', 'mis': 'teddy bear'}
    class_image_dict = defaultdict(list)
    predictions = defaultdict(list)
    for label, img in zip(labels, images):
        output = torch.nn.Softmax()(model(img.unsqueeze(0).cuda()))
        _, pred = torch.max(output, 1)
        predictions[int(label)].append(int(pred))

        if generate_output_images:
            # for creating mean images out of predicted classes
            class_image_dict[int(pred)].append(convert_tensor_to_img(img))

    if generate_output_images:
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
    """
    Compute accuracy for the gven mapping and images
    :param model: model for making predictions
    :param preds_mapping: mapping of neurons to classes
    :param data_dir: directory with images
    :param labels: if provided, accuracy will be performed basing on labels given
    :return:
    """
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
        # calculate predictions for images in the given directory
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

    total = 0
    for img_class, neurons in preds_mapping.items():
        # calculate accuracy basing on predictions made
        acc = 100 * (len(np.where(predictions[img_class] == img_class)[0]) / len(predictions[img_class]))
        print(f'{img_class} -> {acc:.2f}%')
        total += acc
        res[img_class] = f'{acc:.2f}%'
    print(f"Total accuracy: {total / len(preds_mapping.keys()):.2f}%")
    res['total'] = f'{total / len(preds_mapping.keys()):.2f}%'
    return res


def generate_models_comparison(data_dir, parameters):
    """
    Generates accuracy comparison for given models
    :param data_dir: directory with samples
    :param parameters: parameters of the model (path, labels provided, mapping)
    :return:
    """
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


transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = Net(num_classes=num_classes, mnist=False)
model.cuda()

# ten maping jest dobierany na podstawie zdjec wyjsciowych z modelu
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
        'mis': [2, 4],
        'kufel': [1],
        'plyn': [0, 3]
    }
})

# generate_models_comparison(data_dir, paramameters)
