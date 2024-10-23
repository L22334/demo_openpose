import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PoseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        keypoints = []
        for line in lines:
            parts = line.strip().split()
            x = float(parts[0])
            y = float(parts[1])
            confidence = float(parts[2])
            keypoints.append([x, y, confidence])
        keypoints = np.array(keypoints).reshape(-1, 3)  # reshape to (13, 3)
        
        # 只取x坐标和y坐标作为输入，不包括置信度
        input_data = keypoints
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        # 获取标签
        label_file_path = file_path.replace('datasets', 'mylabels')
        with open(label_file_path, 'r') as f:
            lines = f.readlines()

        keypoints = []
        for line in lines:
            parts = line.strip().split()
            x = float(parts[0])
            y = float(parts[1])
            confidence = float(parts[2])
            keypoints.append([x, y, confidence])
        keypoints = np.array(keypoints).reshape(-1, 3)  # reshape to (13, 3)
        target_data = keypoints[:, :2]  # 预测目标也是x和y坐标
        target_tensor = torch.tensor(target_data, dtype=torch.float32)
        # print(input_tensor.shape, target_tensor.shape)
        # print(input_tensor, target_tensor)
        return input_tensor, target_tensor


