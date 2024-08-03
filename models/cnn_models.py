import torch
from torch import nn
import torch.nn.functional as F

num_classes = 36

class ConvSignLangNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv2d(16, 8, 5)
        self.fc1 = nn.Linear(8 * 29 * 29, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x))) # ([32, 8, 29, 29])
        # fully connected layers 
        x = torch.flatten(x, 2) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        return x