import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F
from generateFunc import init, AlexNet, create_dataset_v1, split_dataset
from generateFunc import create_dataset_v1, split_dataset, LeNet, AlexNet, VGG, NiN, ResNet,ResNet_34
import matplotlib.pyplot as plt
# img_transforms=transforms.Compose([
#         transforms.CenterCrop(256),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=0,std=1)
#     ])
#
#
# img_path='./dataset/test/img/epoch_89_img.png'
# img=Image.open(img_path).convert('RGB')
# # img_1=transforms.CenterCrop(256)(img)
# img_1=img_transforms(img)
# print(img_1.shape)
# img_1=img_1.to(device=torch.device('cuda'))
# print(img_1.device)
# img_1=transforms.ToPILImage()(img_1)
#
# img_1.show()
# img.show()
# print(img.size)
# img_1.show()
# img=PIL.Image.open(img_path)
# print(img.size)
# print(img.format)
# print(img.mode)
# img=transforms.ToTensor()(img)
# print(img.shape)
# a = torch.randn(16, 3, 256, 256)
# b = F.conv2d(a, torch.randn(3,1, 64, 64), groups=3)
# print(b.shape)
# a=torch.randn(32,3,256,256)
# layer=nn.Conv2d(3,84,4,4,groups=3)
# b=layer.forward(a)
# print(b.shape)
# print(__name__)
# print(torch.cuda.get_device_name())


pro_r, pro, E_mode, mode_count = init()
model_path = './model/test_ResNet_34_model_v1.pt'
model_data = torch.load(model_path)
model = ResNet_34()
model.load_state_dict(model_data['model_state_dict'])
# for param in model.state_dict():
#     print(param,'\t',model.state_dict()[param].size())
# print(model)
img_dir = './dataset/random_only/img'
doc_dir = './dataset/random_only/doc'

dataset = create_dataset_v1(img_dir, doc_dir)
train_iter, val_iter, test_iter = split_dataset(dataset,
                                                train_size=346801,
                                                val_size=3000,
                                                test_size=200,
                                                batch_size=100)

model.eval()
test_set = next(iter(test_iter))
test_data = test_set['a_fhi'][0]
a = test_data[:42]
print(a)
fhi = test_data[42:]
print(a.sum())
test_img = test_set['img'][0]
# print(test_data.shape)

Nx = 256
Ny = 256
_a = a
_a = _a / torch.sqrt(torch.sum(torch.pow(_a, 2)))
_fhi = fhi
_complex = _a * torch.exp(1j * _fhi)
print(_a.sum())
E_output = torch.zeros(size=(Nx, Ny), device='cuda')

for i in range(mode_count):
    E_output = torch.squeeze(_complex[i] * pro[i] * E_mode[i, :, :]) + E_output

E_output = E_output / torch.sqrt(
    torch.sum(
        torch.pow(torch.abs(E_output), 2)))

I_output = torch.pow(torch.abs(E_output), 2).cpu()
# print(test_data[0].shape)
img = transforms.ToPILImage()(test_img)
# img.show()
# plt.figure()
# plt.imshow(img)
# plt.show()

with torch.no_grad():
    test_output = model(test_img.unsqueeze(0)).reshape(-1, 1)
    # print(test_output.shape)
    test_a = test_output[:42]
    # test_a = test_a / torch.sqrt(torch.sum(torch.pow(test_a, 2)))
    print(test_a)
    test_fhi = test_output[42:]
    # test_a = test_a / torch.sqrt(torch.sum(torch.pow(test_a, 2)))

    test_complex = test_a * torch.exp(1j * test_fhi)
    test_E = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        test_E = torch.squeeze(test_complex[i] * pro[i] * E_mode[i, :, :]) + test_E

    test_E = test_E / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(test_E), 2)))
    test_I = torch.pow(torch.abs(test_E), 2).cpu()
    plt.figure()
    plt.title('output')
    plt.imshow(test_I, cmap='jet')
    plt.show()
    img.show()
    # print(test_output)
    # print(test_data)
    # print(a.sum())
    # print(test_a.sum())
    print(f'debug {F.mse_loss(test_data,test_output)}')

