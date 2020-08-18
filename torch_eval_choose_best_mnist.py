import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from copy import deepcopy
from torchvision import datasets, transforms
from collections import defaultdict

from images_mean import generate_mean_images
from architecture import Net



def convert_tensor_to_img(tensor):
    """
    Converts torch tensor to 3D matrix image
    :param tensor:
    :return image
    """
    # tensor = tensor / 2 + 0.5
    return np.squeeze(np.transpose(tensor.cpu().numpy(), (1, 2, 0)))


def show_img(img):
    """
    Display image out of tensor
    :param img:
    :return:
    """
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))


# only tensor converting
transform = transforms.Compose([
    transforms.ToTensor(),
])

# built in MNST dataset
dataset = datasets.MNIST(root='data', train=True,
                         download=True, transform=transform)

# batch_size = len(dataset) # too much RAM memory needed to use whole MNIST
batch_size = 200
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_classes = 2 * len(dataset.classes)

# implement xavier
# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         torch.nn.init.xavier_uniform_(m.weight.data, gain=0.1)
#         if m.bias is not None:
#             torch.nn.init.zeros_(m.bias)
#     if isinstance(m, nn.Linear):
#         torch.nn.init.uniform_(m.weight, a=0.01, b=0.1)
#         if m.bias is not None:
#             torch.nn.init.zeros_(m.bias)


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
            try:
                # deleting previously used sample to avoid using the same one again
                # necessary because network sometimes was choosing one sample for each output neuron
                # and it was corrupting whole process
                output_to_update = np.delete(output_to_update, np.where(output_to_update == el)[0][0], axis=0)
            except Exception as e:
                print(e)
                continue
    return dataset


# Grid search for finding best model
lrs = [0.001]
epochs = [100, 150]
classes = [10, 15]
labeled = [3, 4]

for num_labeled in labeled:
    for lr in lrs:
        for num_epoch in epochs:
            for num_classes in classes:

                model = Net(num_classes=num_classes, mnist=True)
                # model.apply(weights_init)
                model.cuda()

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # dictionary which is used to store image samples for each output neuron
                class_image_dict = {i: [] for i in range(num_classes)}

                # main loop
                num_of_best = 1  # initial value
                # indicates in which epochs number of samples for each neuron will be increased
                increase_samples_number = np.linspace(3, num_epoch - 1, 50, dtype=int)
                start_time = time.time()
                for epoch in range(1, num_epoch):

                    if epoch in increase_samples_number:
                        num_of_best += 1
                    for batch_number, (data, labels) in enumerate(dataloader):
                        data = data.cuda()
                        optimizer.zero_grad()
                        output = model(data)
                        # for few first batches labeled data are used to lead the learning in correct direction
                        if epoch == 1 and batch_number in list(range(num_labeled)):
                            print('labeled')
                            labels = labels.cuda()
                            loss = criterion(output, labels)
                            loss.backward()
                            optimizer.step()
                        else:
                            # learning without labels, dataset is being created by find_n_best function
                            dataset = find_n_best(output, n=num_of_best, num_classes=num_classes)
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

                            # if epoch in save_conv_layer:
                            #     conv_layer_dict[epoch] = deepcopy(model.conv1)
                            if epoch > num_epoch - 5:
                                # save few last samples in the dictionary to create the figure with mean images
                                for index, tensor in zip(target, data[data_index]):
                                    class_image_dict[int(index)].append(convert_tensor_to_img(tensor))

                    if epoch % 5 == 0:
                        print(f'Epoch: {epoch}')

                # commented out in the MNIST because .pickle file weight was too big
                # with open(f'class_image_dict_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}.pickle', 'wb') as f:
                #     pickle.dump(class_image_dict, f)

                end_time = time.time()
                # generate image from class_image_dict and save it
                generate_mean_images(class_image_dict,
                                     f'img_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}_{num_labeled * 2000}labels.jpg')
                # colab saving, used for training on google colab, not local deelopment
                # copy2(f'img_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}_{num_labeled * 2000}labels.jpg',
                #       'drive/My Drive/magisterka/colab_outputs')

                time_for_epoch = (end_time - start_time) / num_epoch
                # Saving trained model with the information about parameters
                torch.save({'model': model.state_dict(),
                            'time_per_epoch': time_for_epoch,
                            'optimizer': optimizer.defaults},
                           f'model_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}_{num_labeled * 2000}labels.pt')
                # copy2(f'model_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}_{num_labeled * 2000}labels.pt',
                #       'drive/My Drive/magisterka/colab_outputs')

