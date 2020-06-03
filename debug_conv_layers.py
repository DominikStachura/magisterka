import pickle
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt



transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# dobre
# layer_path = 'layers_outputs/new/1/layer_dict_0_01_100_3.pickle'

layer_path = 'layers_outputs/new/2/layer_dict_0_001_200_3.pickle'
layer = pickle.load(open(layer_path, 'rb'))

img = Image.open('datasets/new/kufel/augmented0.jpg')
img = transform(img).unsqueeze(0)
print("")
# plt.imshow(layer[99](img.cuda())[0][0].cpu().detach().numpy())