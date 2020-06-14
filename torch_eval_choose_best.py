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

batch_size = 1


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
batch_size = len(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_classes = 2 * len(dataset.classes)
classes = dataset.classes


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
epochs = [150, 200, 300]
classes = [3, 4, 5, 6, 7]

for lr in lrs:
    for num_epoch in epochs:
        for num_classes in classes:
            try:
                model = Net()
                # model.apply(weights_init)
                model.cuda()




                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                class_image_dict = {i: [] for i in range(num_classes)}
                filters_dict = {}

                # main loop
                num_of_best = 1
                save_conv_layer = np.linspace(1, num_epoch-1, 20, dtype=int)
                increase_samples_number = np.linspace(3, num_epoch-1, 25, dtype=int)
                conv_layer_dict = {}
                for epoch in range(1, num_epoch):
                    # najpierw sprawdzam co wypluje model i na tej podstawie dobieram labele
                    # model.eval()
                    # wewnetrzne fory wykonuja sieraz bo batch size wielkosci calego zbioru
                    for data, _ in dataloader:
                        data = data.cuda()

                        optimizer.zero_grad()
                        output = model(data)
                        if epoch in increase_samples_number:
                            num_of_best += 1
                        dataset = find_n_best(output, n=num_of_best, num_classes=num_classes)
                        # target = torch.argmax(torch.nn.Softmax()(output), dim=1)
                        data_index = []
                        target = []
                        for value in list(dataset.values()):
                            for el in value:
                                data_index.append(el)
                        for el in list(dataset.keys()):
                            for _ in range(num_of_best):
                                target.append(el)

                        # data = data[data_index]
                        target = torch.tensor(target).cuda()

                        # new_output = model
                        loss = criterion(output[data_index], target)

                        loss.backward()

                        optimizer.step()


                        if epoch in save_conv_layer:
                            conv_layer_dict[epoch] = deepcopy(model.conv1)

                        for index, tensor in zip(target, data[data_index]):
                            class_image_dict[int(index)].append(convert_tensor_to_img(tensor))

                        filters_dict[epoch] = model.conv1.weight.data.permute(0, 2, 3, 1).cpu().numpy()

                    # del data
                    # torch.cuda.empty_cache()

                    if epoch % 5 == 0:
                        print(f'Epoch: {epoch}')

                with open(f'pickle_outputs/new/granulacja/class_image_dict_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}.pickle', 'wb') as f:
                    pickle.dump(class_image_dict, f)

                with open(f'layers_outputs/new/granulacja/layer_dict_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}.pickle', 'wb') as f:
                    pickle.dump(conv_layer_dict, f)

                # with open(f'filters_output/new/granulacja/filters_dict_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}.pickle', 'wb') as f:
                #     pickle.dump(filters_dict, f)

                torch.save(model.state_dict(), f'model_outputs/new/granulacja/model_{str(lr).replace(".", "_")}_{num_epoch}_{num_classes}.pt')

                # del data
                # torch.cuda.empty_cache()
                # test, be akualizowania wag
                # model.eval()
                # for epoch in range(1, 30):
                #     for data, _ in dataloader:
                #         data = data.cuda()
                #         output = model(data)
                #         _, pred = torch.max(output, 1)
                #         # pred = torch.max(torch.nn.Softmax(dim=1)(output), 1)
                #         max_output = int(pred.cpu().numpy())
            except Exception as e:
                print(e)
                continue
