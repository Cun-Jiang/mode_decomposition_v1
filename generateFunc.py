import os

import PIL.Image
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
from torchvision import transforms
from torch.nn import functional as F


def init(doc_path='./dataset/init_data.pth'):
    torch.set_default_device('cuda')
    init_data = torch.load(doc_path, map_location=torch.device('cuda:0'))
    pro_r = init_data['pro_r']
    pro = init_data['pro']
    E_mode = init_data['E_mode']
    mode_count = init_data['modeCount']

    return pro_r, pro, E_mode, mode_count


def random_data_img_generate(pro, E_mode, mode_count, epoch=0):
    Nx = 256
    Ny = 256

    _a = torch.rand(size=(mode_count, 1))
    _a = _a / torch.sqrt(torch.sum(torch.pow(_a, 2)))
    _fhi = torch.rand(size=(mode_count, 1)) * 2 * torch.pi
    _complex = _a * torch.exp(1j * _fhi)

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')

    for i in range(mode_count):
        E_output = torch.squeeze(_complex[i] * pro[i] * E_mode[i, :, :]) + E_output

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))

    I_output = torch.pow(torch.abs(E_output), 2).cpu()

    torch.save({
        'a': _a,
        'fhi': _fhi,
        'E_output': E_output,
    }, f'./dataset/random_only/doc/epoch_{epoch}_data.pth')

    plt.figure()
    plt.axis('off')
    plt.imshow(I_output, cmap='jet')
    plt.savefig(f'./dataset/random_only/img/epoch_{epoch}_img.png', bbox_inches='tight', pad_inches=0)
    plt.close()


class create_dataset_v1(data.Dataset):
    def __init__(self, img_dir, doc_dir):
        self.img_dir = img_dir
        self.doc_dir = doc_dir
        self.img_list = os.listdir(img_dir)
        self.doc_list = os.listdir(doc_dir)
        self.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=0, std=1)
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_file = os.path.join(self.img_dir, self.img_list[index])
        doc_file = os.path.join(self.doc_dir, self.doc_list[index])
        # print(img_file)
        # print(doc_file)
        # print('-------')

        img = PIL.Image.open(img_file).convert('RGB')
        img = self.transform(img)
        img = img.to(device=torch.device('cuda'))

        doc = torch.load(doc_file, map_location=torch.device('cuda'))
        a = doc['a']
        fhi = doc['fhi']
        a_fhi = torch.cat([a, fhi])
        a_and_cos_fhi = torch.cat([a, torch.cos(fhi)])
        a_exp_fhi = a * torch.exp(fhi)
        a_cos_fhi = a * torch.cos(fhi)
        E_output = doc['E_output']

        return {'img': img,
                'a': a,
                'fhi': fhi,
                'a_fhi': a_fhi,
                'a_and_cos_fhi': a_and_cos_fhi,
                'a_exp_fhi': a_exp_fhi,
                'a_cos_fhi': a_cos_fhi,
                'E_output': E_output}


def split_dataset(dataset, train_size, val_size, test_size, batch_size=32, is_shuffle=True):
    train_set, val_set, test_set = data.random_split(dataset,
                                                     [train_size, val_size, test_size],
                                                     torch.Generator(device='cuda'))
    train_iter = data.DataLoader(train_set, batch_size,
                                 shuffle=is_shuffle,
                                 generator=torch.Generator(device='cuda'),
                                 # num_workers=2,
                                 # pin_memory=True,
                                 # prefetch_factor=1
                                 )
    val_iter = data.DataLoader(val_set, batch_size,
                               shuffle=is_shuffle,
                               generator=torch.Generator(device='cuda'),
                               # num_workers=2,
                               # pin_memory=True
                               )
    test_iter = data.DataLoader(test_set, batch_size,
                                shuffle=is_shuffle,
                                generator=torch.Generator(device='cuda'),
                                # num_workers=2,
                                # pin_memory=True
                                )
    return train_iter, val_iter, test_iter


class LeNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_1 = nn.Conv2d(3, 84, 2, 2)  # -> 84 * 128 * 128
        # relu
        self.pool_1 = nn.MaxPool2d(2, 2)  # -> 84 * 64 * 64

        self.conv_2 = nn.Conv2d(84, 42, 2, 2)  # -> 42 * 32 * 32
        # relu
        self.pool_2 = nn.MaxPool2d(2, 2)  # -> 42 * 16 * 16

        # flatten -> 10752

        self.linear_1 = nn.Linear(10752, 4096)
        # relu
        self.linear_2 = nn.Linear(4096, 256)
        # relu
        self.linear_3 = nn.Linear(256, 84)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.pool_2(x)

        x = torch.flatten(x, 1)

        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)

        return x


class AlexNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_1 = nn.Conv2d(3, 84, 11, 4, 2)  # -> 84 * 63 * 63
        self.pool_1 = nn.MaxPool2d(3, 2)  # -> 84 * 31 * 31

        self.conv_2 = nn.Conv2d(84, 256, 5, 1, 2)  # -> 256 * 31 * 31
        self.pool_2 = nn.MaxPool2d(3, 2)  # -> 256 * 15 * 15

        self.relu = nn.ReLU()

        self.sequential_1 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(),  # -> 256 * 15 * 15
            nn.MaxPool2d(3, 2),  # -> 256 * 7 * 7
            nn.Flatten(),  # -> 12544
        )

        self.sequential_2 = nn.Sequential(
            nn.Linear(12544, 8192), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8192, 8192), nn.ReLU(),
            nn.Dropout(),

            nn.Linear(8192, 84)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.pool_2(x)

        x = self.sequential_1(x)
        x = self.sequential_2(x)

        return x


def vgg_block(num_conv, in_channels, out_channel):
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channel
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# def __VGG(conv_arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):
#     conv_blocks = []
#     in_channel = 3
#     for (num_conv, out_channel) in conv_arch:
#         conv_blocks.append(vgg_block(num_conv, in_channel, out_channel))
#         in_channel = out_channel
#
#     return nn.Sequential(
#         *conv_blocks,
#         nn.Flatten(),
#         nn.Linear(out_channel * 8 * 8, 4096), nn.ReLU(), nn.Dropout(),
#         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
#         nn.Linear(4096, 84)
#     )


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = self.__VGG()

    def forward(self, x):
        x = self.net(x)

        return x

    @staticmethod
    def __VGG(conv_arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):
        conv_blocks = []
        in_channel = 3
        for (num_conv, out_channel) in conv_arch:
            conv_blocks.append(vgg_block(num_conv, in_channel, out_channel))
            in_channel = out_channel

        return nn.Sequential(
            *conv_blocks,
            nn.Flatten(),
            nn.Linear(out_channel * 8 * 8, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 84)
        )


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


# def __NiN():
#     return nn.Sequential(
#         nin_block(3, 96, 11, 4, 0),
#         nn.MaxPool2d(3, 2),
#         nin_block(96, 256, 5, 1, 2),
#         nn.MaxPool2d(3, 2),
#         nin_block(256, 384, 3, 1, 1),
#         nn.MaxPool2d(3, 2),
#         nn.Dropout(),
#         nin_block(384, 84, 3, 1, 1),
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Flatten()
#     )


class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = self.__NiN()

    def forward(self, x):
        x = self.net(x)

        return x

    @staticmethod
    def __NiN():
        return nn.Sequential(
            nin_block(3, 96, 11, 4, 0),
            nn.MaxPool2d(3, 2),
            nin_block(96, 256, 5, 1, 2),
            nn.MaxPool2d(3, 2),
            nin_block(256, 384, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            nn.Dropout(),
            nin_block(384, 84, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv_3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv_3 = None

        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn_1(self.conv_1(x)))
        y = self.bn_2(self.conv_2(y))
        # print(f'before x:{x.shape}, y:{y.shape}')
        if self.conv_3:
            x = self.conv_3(x)
        y += x
        return F.relu(y)


def resnet_block(in_channel, out_channel, num_residuals, first_block=False):
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.append(Residual(in_channel, out_channel, use_1x1conv=True, strides=2))
        else:
            block.append(Residual(out_channel, out_channel))
    return block


# def __ResNet():
#     b_1 = nn.Sequential(
#         nn.Conv2d(3, 64, 2, 2, 3),
#         nn.BatchNorm2d(64),
#         nn.ReLU(),
#         nn.MaxPool2d(3, 2, 1)
#     )
#     b_2 = nn.Sequential(
#         *resnet_block(64, 64, 2, first_block=True)
#     )
#     b_3 = nn.Sequential(
#         *resnet_block(64, 128, 2)
#     )
#     b_4 = nn.Sequential(
#         *resnet_block(128, 256, 2)
#     )
#     b_5 = nn.Sequential(
#         *resnet_block(256, 512, 2)
#     )
#     return nn.Sequential(
#         b_1, b_2, b_3, b_4, b_5,
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Flatten(),
#         nn.Linear(512, 84)
#     )


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = self.__ResNet()

    def forward(self, x):
        x = self.net(x)

        return x

    @staticmethod
    def __ResNet():
        b_1 = nn.Sequential(
            nn.Conv2d(3, 64, 2, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        b_2 = nn.Sequential(
            *resnet_block(64, 64, 2, first_block=True)
        )
        b_3 = nn.Sequential(
            *resnet_block(64, 128, 3)
        )
        b_4 = nn.Sequential(
            *resnet_block(128, 256, 4)
        )
        b_5 = nn.Sequential(
            *resnet_block(256, 512, 2)
        )
        return nn.Sequential(
            b_1, b_2, b_3, b_4, b_5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 84)
        )


class ResNet_34(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = self.__ResNet_34()

    def forward(self, x):
        x = self.net(x)

        return x

    @staticmethod
    def __ResNet_34():
        conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        conv_2 = nn.Sequential(
            *resnet_block(64, 64, 3, first_block=True)
        )
        conv_3 = nn.Sequential(
            *resnet_block(64, 128, 4)
        )
        conv_4 = nn.Sequential(
            *resnet_block(128, 256, 6)
        )
        conv_5 = nn.Sequential(
            *resnet_block(256, 512, 3)
        )
        return nn.Sequential(
            conv_1, conv_2, conv_3, conv_4, conv_5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 84)
        )


class Bottleneck(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_shortcuts=True, strides=1, is_bias=False):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, middle_channels, kernel_size=1, padding=0, stride=1, bias=is_bias)
        self.conv_2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1, stride=strides,
                                bias=is_bias)
        self.conv_3 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=is_bias)
        if is_shortcuts is False:
            self.shortcuts = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=strides,
                                       bias=is_bias)
            # self.shortcuts_bn=nn.BatchNorm2d(out_channels)
        else:
            self.shortcuts = None

        self.bn_1 = nn.BatchNorm2d(middle_channels)
        self.bn_2 = nn.BatchNorm2d(middle_channels)
        self.bn_3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        # self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn_1(self.conv_1(x)))
        y = self.relu(self.bn_2(self.conv_2(y)))
        y = self.relu(self.bn_3(self.conv_3(y)))
        if self.shortcuts is not None:
            # y += self.bn_3(self.shortcuts(x))
            y = torch.add(y, self.bn_3(self.shortcuts(x)))
        else:
            # y += x
            y = torch.add(y, x)
        return self.relu(y)


def bottleneck_block(in_channels, middle_channel, out_channels, block_num, is_first_block=False):
    block = []
    for i in range(block_num):
        if i == 0 and not is_first_block:
            block.append(Bottleneck(in_channels, middle_channel, out_channels, False, 2))
        elif i == 0 and is_first_block:
            block.append(Bottleneck(in_channels, middle_channel, out_channels, False))
        else:
            block.append(Bottleneck(out_channels, middle_channel, out_channels, True))
    return block


class deeper_ResNet_101(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = self.__deeper_ResNet()

    def forward(self, x):
        x = self.net(x)
        return x

    @staticmethod
    def __deeper_ResNet():
        block_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            # nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            # nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        block_2 = nn.Sequential(
            *bottleneck_block(64, 64, 256, 3, True)
        )
        block_3 = nn.Sequential(
            *bottleneck_block(256, 128, 512, 4)
        )
        block_4 = nn.Sequential(
            *bottleneck_block(512, 256, 1024, 23)
        )
        block_5 = nn.Sequential(
            *bottleneck_block(1024, 512, 2048, 3)
        )

        return nn.Sequential(
            block_1,
            block_2,
            block_3,
            block_4,
            block_5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 84)
        )
