import torch
from torch import nn, optim
import time
import matplotlib.pyplot as plt
from generateFunc import create_dataset_v1, split_dataset, LeNet, AlexNet, VGG, NiN, ResNet, ResNet_34,deeper_ResNet_101

# torch.set_default_device('cuda')
# img_dir = './dataset/random_only/img'
# doc_dir = './dataset/random_only/doc'
#
# dataset = create_dataset_v1(img_dir, doc_dir)
# train_iter, val_iter, test_iter = split_dataset(dataset,
#                                                 train_size=346001,
#                                                 val_size=3000,
#                                                 test_size=1000,
#                                                 batch_size=512)
#
# model = LeNet()
# loss = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# val_loss_array = []
# test_loss_array = []
#
# # epoch_num = 1000
# epoch_num = 3
# for epoch in range(epoch_num):
#     begin = time.time()
#     model.train()
#     for __iter in train_iter:
#         train_output = model(__iter['img'])
#         train_loss = loss(torch.unsqueeze(train_output, 2), __iter['a_fhi'])
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()
#
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for __val_iter in val_iter:
#             val_output = model(__val_iter['img'])
#             val_loss = loss(torch.unsqueeze(val_output, 2), __val_iter['a_fhi']).item()
#             val_loss_array.append(val_loss)
#
#     end = time.time()
#     # if epoch % 100 == 0:
#     if epoch % 1 == 0:
#         print(f'epoch: {epoch}, loss: {val_loss}, time: {end - begin} s')
#
# model_path = './model/test_model_v1.pt'
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'model': model
# }, model_path)

if __name__ == '__main__':
    torch.set_default_device('cuda')
    img_dir = './dataset/random_only/img'
    doc_dir = './dataset/random_only/doc'

    dataset = create_dataset_v1(img_dir, doc_dir)
    train_iter, val_iter, test_iter = split_dataset(dataset,
                                                    train_size=346801,
                                                    val_size=3000,
                                                    test_size=200,
                                                    batch_size=384)

    # model = AlexNet()
    # model = VGG()
    model = ResNet_34()
    # model = NiN()
    # model = deeper_ResNet_101()
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    val_loss_array = []
    test_loss_array = []

    # epoch_num = 1000
    epoch_num = 20
    # epoch_num = 5
    for epoch in range(epoch_num):
        begin = time.time()
        model.train()
        for __iter in train_iter:
            # print(__iter['img'].shape)
            train_output = model(__iter['img'])
            # print(train_output.shape)
            # print(__iter['a_exp_fhi'].shape)
            # print(torch.unsqueeze(train_output[:, :42] * torch.exp(train_output[:, 42:]), 2).shape)
            # train_loss = loss(torch.unsqueeze(train_output, 2), __iter['a_fhi'])
            # train_loss = loss(torch.unsqueeze(train_output[:, :42] * torch.exp(train_output[:, 42:]), 2),
            #                   __iter['a_exp_fhi'])
            train_loss = loss(torch.unsqueeze(train_output, 2), __iter['a_and_cos_fhi'])
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for __val_iter in val_iter:
                val_output = model(__val_iter['img'])
                val_loss = loss(torch.unsqueeze(val_output, 2), __val_iter['a_fhi']).item()
                val_loss_array.append(val_loss)

        end = time.time()
        # if epoch % 100 == 0:
        if epoch % 1 == 0:
            print(f'epoch: {epoch}, loss: {val_loss}, time: {end - begin} s')

    model_path = 'model/ResNet_34_model_v1.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model': model
    }, model_path)

    plt.figure()
    plt.title('loss')
    plt.plot(val_loss_array)
    plt.show()
