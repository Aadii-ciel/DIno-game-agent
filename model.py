import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        # Input: 4x84x84 (stacked frames)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the flattened features
        # (84-8)/4 + 1 = 20
        # (20-4)/2 + 1 = 9
        # (9-3)/1 + 1 = 7
        # 7 * 7 * 64 = 3136
        
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        # x shape: (batch_size, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
