import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from copy import deepcopy
from torchvision import datasets, transforms
from collections import defaultdict

from architecture import Net

batch_size = 1


def convert_tensor_to_img(tensor):
    """
    Converts torch tensor to 3D matrix image
    :param tensor:
    :return image
    """
    tensor = tensor.squeeze()
    # tensor = tensor / 2 + 0.5
    return np.transpose(tensor.cpu().numpy(), (1, 2, 0))


def show_img(img):
    """
    Display image out of tensor
    :param img:
    :return:
    """
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
batch_size = len(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# implement xavier
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def find_n_best(output, n=1, num_classes=6):
    """
    function takes outputs from whole dataset or big batches forward pass and choose samples
    that strengthen each neuron most
    :param output: output from the network
    :param n: number of samples for each class
    :param num_classes: number of classes to which function will assign samples
    :return: new dataset with best samples chosen for each class
    """
    # {target0: [zdj1, zdj2...], target1:[...]}
    dataset = defaultdict(list)
    output = output.cpu().detach().numpy()
    output_to_update = deepcopy(output)
    for i in range(num_classes):
        # choose n best samples basing on the maximum value on i th vector position
        best = sorted(output_to_update, key=lambda x: x[i])[::-1][:n]
        for el in best:
            # append index of the best sample on the i th position
            dataset[i].append(np.where(output == el)[0][0])
            # deleting previously used sample to avoid using the same one again
            # necessary because network sometimes was choosing one sample for each output neuron
            # and it was corrupting whole process
            output_to_update = np.delete(output_to_update, np.where(output_to_update == el)[0][0], axis=0)
    return dataset


# Grid search for finding best model
lrs = [0.001]
epochs = [80, 200]
classes = [3, 5]
xavier = [True, False]
opts = [optim.Adam, optim.Adadelta, optim.Adagrad]

for opt in opts:
    for lr in lrs:
        for num_epoch in epochs:
            for num_classes in classes:
                try:
                    model = Net(num_classes=num_classes)
                    if xavier:
                        model.apply(weights_init)
                    model.cuda()

                    criterion = nn.CrossEntropyLoss()
                    # optimizer = optim.Adam(model.parameters(), lr=lr)
                    # trying different optimizers
                    optimizer = opt(model.parameters())

                    # dictionary which is used to store image samples for each output neuron
                    class_image_dict = {i: [] for i in range(num_classes)}
                    filters_dict = {}

                    # main loop
                    num_of_best = 1
                    # was used for saving some of convulutional layers
                    save_conv_layer = np.linspace(1, num_epoch - 1, 20, dtype=int)
                    # indicates in which epochs number of samples for each neuron will be increased
                    increase_samples_number = np.linspace(3, num_epoch - 1, 25, dtype=int)
                    conv_layer_dict = {}
                    start_time = time.time()
                    for epoch in range(1, num_epoch):
                        for data, _ in dataloader:
                            data = data.cuda()

                            optimizer.zero_grad()
                            output = model(data)
                            if epoch in increase_samples_number:
                                num_of_best += 1
                            # learning without labels, dataset is being created by find_n_best function
                            dataset = find_n_best(output, n=num_of_best, num_classes=num_classes)
                            # target = torch.argmax(torch.nn.Softmax()(output), dim=1)
                            data_index = []
                            target = []
                            for value in list(dataset.values()):
                                for el in value:
                                    # indexes of samples that will be used for training
                                    data_index.append(el)
                            for el in list(dataset.keys()):
                                # labels of samples that will be used for training
                                for _ in range(num_of_best):
                                    target.append(el)

                            target = torch.tensor(target).cuda()
                            loss = criterion(output[data_index], target)
                            loss.backward()
                            optimizer.step()

                            for index, tensor in zip(target, data[data_index]):
                                class_image_dict[int(index)].append(convert_tensor_to_img(tensor))

                        if epoch % 5 == 0:
                            print(f'Epoch: {epoch}')

                    with open(
                            f'pickle_outputs/new/parameters/class_image_dict_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}_{type(optimizer).__name__}_xavier_{str(xavier)}.pickle',
                            'wb') as f:
                        pickle.dump(class_image_dict, f)

                    # with open(f'layers_outputs/new/granulacja/layer_dict_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}.pickle', 'wb') as f:
                    #     pickle.dump(conv_layer_dict, f)

                    # with open(f'filters_output/new/granulacja/filters_dict_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}.pickle', 'wb') as f:
                    #     pickle.dump(filters_dict, f)

                    end_time = time.time()
                    time_for_epoch = (end_time - start_time) / num_epoch
                    torch.save({'model': model.state_dict(),
                                'time_per_epoch': time_for_epoch,
                                'optimizer': optimizer.defaults},
                               f'model_outputs/new/parameters/model_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}_{type(optimizer).__name__}_xavier_{str(xavier)}.pt')
                except Exception as e:
                    print(e)
                    continue
