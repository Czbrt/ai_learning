import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import loader as ml


def train_gender_model(model, batch_size, epochs, use_cuda, learning_rate=0.01, weight_decay=1e-3):
    if not os.path.exists('output'):
        os.makedirs('output')

    tra_data = ml.GenderDataset(txt_file='/Users/sunyuliang/Desktop/CV/Python/ML Lab/gender_recognition/sex_train.txt')
    val_data = ml.GenderDataset(txt_file='/Users/sunyuliang/Desktop/CV/Python/ML Lab/gender_recognition/sex_val.txt')
    train_loader = DataLoader(dataset=tra_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)

    if use_cuda:
        print('Training with CUDA')
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30], 0.1)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0

        for batch, (batch_data, batch_label) in enumerate(train_loader):
            loss, train_correct = process_batch(model, batch_data, batch_label, loss_func, use_cuda)
            train_loss += loss
            train_acc += train_correct

            print(f'Epoch: {epoch + 1}/{epochs} Batch: {batch + 1}/{len(train_loader)} '
                  f'Train Loss: {loss:.3f}, Acc: {train_correct / batch_data.size(0):.3f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f'Train Loss: {train_loss / len(train_loader):.6f}, Acc: {train_acc / len(tra_data):.3f}')

        # Evaluation phase
        model.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for batch_data, batch_label in val_loader:
                loss, num_correct = process_batch(model, batch_data, batch_label, loss_func, use_cuda)
                eval_loss += loss
                eval_acc += num_correct

        print(f'Val Loss: {eval_loss / len(val_loader):.6f}, Acc: {eval_acc / len(val_data):.3f}')

        # 保存模型参数
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'output/gender_model_epoch_{epoch + 1}.pth')


def process_batch(model, batch_data, batch_label, loss_func, use_cuda):
    if use_cuda:
        batch_data = batch_data.float().cuda()
        batch_label = batch_label.long().cuda()
    else:
        batch_data = batch_data.float()
        batch_label = batch_label.long()

    out = model(batch_data)
    loss = loss_func(out, batch_label)
    pred = torch.max(out, 1)[1]
    train_correct = (pred == batch_label).sum().item()  # 这里得到的是一个整数

    return loss, train_correct  # 直接返回 loss 和 train_correct


batch_size = 10
epochs = 100
use_cuda = torch.cuda.is_available()


class GenderClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GenderClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # 第一个隐藏层激活
        x = self.fc2(x)  # 输出层
        return x


input_size = 2  # 假设有两个特征（根据你的数据）
hidden_size = 16  # 隐藏层节点数
output_size = 2  # 输出性别类别（例如：0 和 1）

model = GenderClassificationModel(input_size, hidden_size, output_size)

# 调用训练函数
train_gender_model(model, batch_size, epochs, use_cuda)
