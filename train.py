import torch.optim as optim
import torch.nn as nn
import torch
from lib.model import PosePredictor
from lib.datasets import PoseDataset, DataLoader
# 示例调用
train_dataset = PoseDataset('labels/datasets/train2017')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 构建关键点预测模型
model = PosePredictor()
# 使用均方误差损失函数
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# # 设置 MultiStepLR 学习率调度器
# milestones = [10, 20]  # 设定在这些 epoch 时调整学习率
# gamma = 0.1  # 学习率衰减因子
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

# 训练循环
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # 学习率调整
            # scheduler.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")

# 示例调用
train(model, train_loader, optimizer, criterion, epochs=30)
# 保存模型
torch.save(model.state_dict(), 'pose_predictor_epoch30.pth')
