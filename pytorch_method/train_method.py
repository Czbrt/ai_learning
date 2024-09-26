import os
import loader_method as ml
import torch.nn as nn
import torch
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms


def train(model, data_path, batch_size, epoch, use_cuda, learning_rate=0.01, weight_decay=1e-3):
    os.makedirs('./output', exist_ok=True)
    # os.makedirs 创建一个名为 output 的目录，如果已经存在则不抛出异常。
    if True:  # not os.path.exists('output/total.txt'):
        ml.image_list(data_path, 'output/total.txt')
        # 从指定的数据路径生成一个包含图像文件和标签的列表文件。
        ml.shuffle_split('output/total.txt', 'output/train.txt', 'output/val.txt')

    train_data = ml.MyDataset(txt='output/train.txt', transform=transforms.ToTensor())
    val_data = ml.MyDataset(txt='output/val.txt', transform=transforms.ToTensor())
    # transforms.ToTensor() 将图像数据转换为 PyTorch 的张量格式。
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
    # DataLoader 将数据集包装为可迭代的数据加载器，支持批处理和打乱数据。

    if use_cuda:
        print('training with cuda')
        model.cuda()
        # 检查是否可以使用 CUDA（GPU 加速），如果可以则将模型移动到 GPU

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 初始化 Adam 优化器，设置学习率和权重衰减
    # Adam自动调整学习率
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30], 0.1)
    # 在特定的训练轮次（20 和 30）调整学习率，变为原来的10%（0.1）
    loss_func = nn.CrossEntropyLoss()
    # 交叉熵损失函数

    for epoch in range(epoch):
        # training-----------------------------------
        model.train()
        train_loss = 0
        train_acc = 0
        for batch, (batch_data, batch_label) in enumerate(train_loader):
            # 使用 enumerate 来迭代训练数据加载器（train_loader），每次获取一个批次的数据和标签。
            # 迭代时自动进行getitem方法，将img和label返回给batch_data和batch_label
            # batch_data 是当前批次的输入数据，通常是一个张量，包含多个样本
            # batch_label 是当前批次的标签，也通常是一个张量，包含与 batch_data 对应的标签

            loss, train_correct = process_batch(model, batch_data, batch_label, loss_func, use_cuda)
            train_loss += loss.item()
            # 将当前批次的损失添加到总损失中。
            train_acc += train_correct.item()
            # 将正确预测的数量累加到总准确率中。

            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                  % (epoch + 1, epoch, batch, math.ceil(len(train_data) / batch_size),
                     loss.item(), train_correct.item() / batch_data.size(0)))

            optimizer.zero_grad()
            # 在进行反向传播之前，先将优化器中的梯度清零。
            loss.backward()
            # 反向传播
            optimizer.step()
            # 更新参数

        scheduler.step()  # 更新learning rate
        print('Train Loss: %.6f, Acc: %.3f' % (
            train_loss / (math.ceil(len(train_data) / batch_size)), train_acc / (len(train_data))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_data, batch_label in val_loader:

            loss, num_correct = process_batch(model, batch_data, batch_label, loss_func, use_cuda)
            eval_loss += loss.item()
            eval_acc += num_correct.item()

        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (math.ceil(len(val_data) / batch_size)), eval_acc / (len(val_data))))
        # 保存模型。每隔多少帧存模型，此处可修改------------
        if (epoch + 1) % 10 == 0:
            # torch.save(model, 'output/model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')
        # 每个 epoch 保存一次模型参数，方便后续加载和继续训练


def process_batch(model, batch_data, batch_label, loss_func, use_cuda):
    if use_cuda:
        batch_data, batch_label = Variable(batch_data.cuda()), Variable(batch_label.cuda())
    else:
        batch_data, batch_label = Variable(batch_data), Variable(batch_label)

    # 前向传播
    out = model(batch_data)

    # 计算损失
    loss = loss_func(out, batch_label)

    # 预测类别
    pred = torch.max(out, 1)[1]
    train_correct = (pred == batch_label).sum().item()

    return loss.item(), train_correct.item()
