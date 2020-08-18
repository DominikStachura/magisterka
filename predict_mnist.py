import torch
from torchvision import transforms
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torchvision import datasets
import pickle
from images_mean import generate_mean_images
from architecture import Net


"""
Similar to predict.py
"""

# MODEL = Path('model_outputs/new/2/model_0_001_200_3.pt')
MODEL = Path('model_outputs/mnist_labels/model_0_001_80_10_8000labels.pt')
num_classes = int(MODEL.stem.split('_')[-2])


transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.MNIST(root='data', train=True,
                         download=True, transform=transform)


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
        generate_mean_images(class_image_dict, f'images/{output_images_name}_generated')

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





model = Net(num_classes=num_classes, mnist=True)
model.cuda()

# ten maping jest dobierany na podstawie zdjec wyjsciowych z modelu
preds_mapping = {
    # 0: [5],
    1: [5],
    2: [2],
    3: [0],
    4: [6],
    5: [3],
    6: [1],
    7: [7],
    8: [8],
    9: [9]
}

model.load_state_dict(torch.load(MODEL)['model'])


batch_size = 30000
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

classes = dataset.classes
dataiter = iter(dataloader)
images, labels = dataiter.next()

# histograms

# plot_class_histograms(model, images, labels, num_classes=num_classes, output_images_name=MODEL.stem)
#
# create_class_imgs_dict(model, images)
compute_accuracy(model, images, labels, preds_mapping)

