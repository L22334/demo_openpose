import torch.nn as nn
import torch.nn.functional as F

class PosePredictor(nn.Module):
    def __init__(self):
        super(PosePredictor, self).__init__()
        self.fc1 = nn.Linear(39, 64)  # 输入维度为13个关键点的x和y坐标
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 26)  # 输出维度为13个关键点的x和y坐标
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(x.size(0), 13, 2)  # 将输出重新调整为13个关键点的x和y坐标
        return x


