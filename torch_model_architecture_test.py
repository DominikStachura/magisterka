import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size = 2





# loading data
data_dir = 'datasets'
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def show_img(img):
    img = img * 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))


def visualize_filters(filters):
    for idx, filter in enumerate(filters):
        plt.subplot(4, 8, idx+1)
        show_img(filter)
    plt.show()

def visualize_conv1_output(conv):
    for idx, filter in enumerate(conv):
        plt.subplot(4, 8, idx+1)
        plt.imshow(filter.detach().cpu().numpy())
    plt.show()

dataset = datasets.ImageFolder(data_dir, transform=transform)
batch_size = len(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_classes = len(dataset.classes)
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
    def __init__(self, big_filters=False):
        super(Net, self).__init__()
        if not big_filters:
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

        else:
            self.conv1 = nn.Conv2d(3, 32, 11, padding=5)
            self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
            self.maxpool = nn.MaxPool2d(2, 2)
            # 2048 na wyjsciu z conv
            self.fc1 = nn.Linear(4 * 4 * 128, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.maxpool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net(big_filters=True)
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#
for epoch in range(1, 20):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()
    for data, target in dataloader:
        # move tensors to GPU if CUDA is available
        # if epoch%4==0:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2
        data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)
    if epoch == 9:
        # filters = model.conv1.weight.data.permute(0, 2, 3, 1).cpu().numpy()
        # filters = model.conv1.weight.data.cpu().numpy()
        # visualize_filters(filters)
        # visualize_conv1_output(model.conv1(data)[0])
        pass


    if epoch % 2 == 0:
        print(f'Epoch: {epoch}, Loss: {train_loss}')
#
torch.save(model.state_dict(), 'model_test.pt')

# model.load_state_dict(torch.load('model_test.pt'))
# #
# idx_to_class = dict(enumerate(dataset.classes))
# from PIL import Image
# img = Image.open('mis_z_internetu.png')
# img = transform(img).unsqueeze(0)
# output = model(img.cuda())
# _, pred = torch.max(output, 1)
# print(idx_to_class[int(pred)])
# print(dataset.class_to_idx)

test_loss = 0.0
class_correct = list(0. for i in range(3))
class_total = list(0. for i in range(3))

model.eval()
# iterate over test data
for data, target in dataloader:
    # move tensors to GPU if CUDA is available

    data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(dataloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(3):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
