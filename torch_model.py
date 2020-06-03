import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

batch_size = 1

def convert_tensor_to_img(tensor):
    tensor = tensor.squeeze()
    tensor = tensor / 2 + 0.5
    return np.transpose(tensor.cpu().numpy(), (1, 2, 0))
def show_img(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))


# loading data
data_dir = 'datasets'
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
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
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
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

model = Net()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)


class_image_dict = {i: [] for i in range(num_classes)}
for epoch in range(1, 30):
    model.train()
    for data, _ in dataloader:
        # move tensors to GPU if CUDA is available
        data = data.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        _, pred = torch.max(output, 1)
        max_output = int(pred.cpu().numpy())
        class_image_dict[max_output].append(convert_tensor_to_img(data))
        target = pred
        loss = criterion(output, target)
        # # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # # perform a single optimization step (parameter update)
        optimizer.step()
        # # update training loss
        # train_loss += loss.item() * data.size(0)

    if epoch % 5 == 0:
        print(f'Epoch: {epoch}')

with open('class_image_dict.pickle', 'wb') as f:
    pickle.dump(class_image_dict, f)

torch.save(model.state_dict(), 'model_test.pt')


# test, be akualizowania wag
model.eval()
for epoch in range(1, 30):
    for data, _ in dataloader:
        data = data.cuda()
        output = model(data)
        _, pred = torch.max(output, 1)
        # pred = torch.max(torch.nn.Softmax(dim=1)(output), 1)
        max_output = int(pred.cpu().numpy())

