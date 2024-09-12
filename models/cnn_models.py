import torch
from torch import nn
import torch.nn.functional as F

_num_classes = 36

class ConvSignLangNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.fc1 = nn.Linear(8*29*29, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, _num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # ([32, 8, 29, 29])

        # fully connected layers 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

class ConvSignLangNN_7(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.fc1 = nn.Linear(8*29*29, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 84)
        self.fc6 = nn.Linear(84, _num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # ([32, 8, 29, 29])

        # fully connected layers 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)       
        return x



class ConvSignLangNN_6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.fc1 = nn.Linear(8*29*29, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 84)
        self.fc5 = nn.Linear(84, _num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # ([32, 8, 29, 29])
        # fully connected layers 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)       

        return x

class ConvSignLangNN_7_(nn.Module):
    def __init__(self, 
                 first_dim: int, 
                 second_dim: int, 
                 third_dim: int, 
                 fourth_dim: int,
                 fifth_dim: int,
                 conv1_in: int,
                 conv2_in: int):
        super().__init__()
        self.conv1 = nn.Conv2d(conv1_in, conv2_in, kernel_size=4, )
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(conv2_in, 8, 5)
        self.fc1 = nn.Linear(8*29*29, first_dim)
        self.fc2 = nn.Linear(first_dim, second_dim)
        self.fc3 = nn.Linear(second_dim,third_dim)
        self.fc4 = nn.Linear(third_dim, fourth_dim,)
        self.fc5 = nn.Linear(fourth_dim, fifth_dim)
        self.fc6 = nn.Linear(fifth_dim, _num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # ([32, 8, 29, 29])
        # fully connected layers 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)       
        return x


class ConvSignLangNN_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.fc1 = nn.Linear(8*29*29, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 84)
        self.fc4 = nn.Linear(84, _num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # ([32, 8, 29, 29])
        # fully connected layers 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class SingleConvSignLang_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.fc1 = nn.Linear(8*29*29, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 84)
        self.fc4 = nn.Linear(84, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # ([32, 8, 29, 29])
        # fully connected layers 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))

        return x


class ConvSignLangNN_4_(nn.Module):
    def __init__(self, 
                 first_dim: int, 
                 second_dim: int, 
                 third_dim: int, 
                 conv1_in: int,
                 conv2_in: int,
                 conv3_in: int):
        super().__init__()
        self.conv1 = nn.Conv2d(conv1_in, conv2_in, kernel_size=3, )
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(conv2_in, conv3_in, 3)
        self.conv3 = nn.Conv2d(conv3_in, 24, 3)
        self.fc1 = nn.Linear(3456, first_dim)
        self.fc2 = nn.Linear(first_dim, second_dim)
        self.fc3 = nn.Linear(second_dim,third_dim)
        self.fc4 = nn.Linear(third_dim, _num_classes,)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        # fully connected layers 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



"""
             ch,   w,   h
Input Shape: [3, 128, 128])
After conv1: [8, 31, 31])
After conv2: [16, 29, 29])
After conv3: [8, 6, 6])
"""
class ConvSignLangNN_5_(nn.Module):
    def __init__(self, 
                 first_dim: int, 
                 second_dim: int, 
                 third_dim: int, 
                 fourth_dim: int, 
                 conv1_in: int,
                 conv2_in: int,
                 conv3_in: int):
        super().__init__()
        self.conv1 = nn.Conv2d(conv1_in, conv2_in, kernel_size=3, )
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(conv2_in, conv3_in, 3)
        self.conv3 = nn.Conv2d(conv3_in, 8, 3)
        self.fc1 = nn.Linear(8 * 6 * 6, first_dim)
        self.fc2 = nn.Linear(first_dim, second_dim)
        self.fc3 = nn.Linear(second_dim,third_dim)
        self.fc4 = nn.Linear(third_dim, fourth_dim)
        self.fc5 = nn.Linear(fourth_dim, _num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x)) 
        x = self.pool(F.relu(self.conv3(x))) 
        # fully connected layers 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class ConvSignLangNN_4_4(nn.Module):
    def __init__(self, 
                 first_dim: int, 
                 second_dim: int, 
                 third_dim: int, 
                 conv1_in: int,
                 conv2_in: int,
                 conv3_in: int,
                 conv4_in: int):
        super().__init__()
        self.conv1 = nn.Conv2d(conv1_in, conv2_in, kernel_size=3, )
        self.pool3 = nn.MaxPool2d(3, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv2_in, conv3_in, 3)
        self.conv3 = nn.Conv2d(conv3_in, conv4_in, 3)
        self.conv4 = nn.Conv2d(conv4_in, 24, 3)
        self.fc1 = nn.Linear(1944, first_dim)
        self.fc2 = nn.Linear(first_dim, second_dim)
        self.fc3 = nn.Linear(second_dim,third_dim)
        self.fc4 = nn.Linear(third_dim, _num_classes,)
    
    def forward(self, x):
        x = self.pool2(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool3(F.relu(self.conv4(x)))
        # fully connected layers 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ConvSignLangNN_4_(nn.Module):
    def __init__(self, 
                 first_dim: int, 
                 second_dim: int, 
                 third_dim: int, 
                 conv1_in: int,
                 conv2_in: int,
                 conv3_in: int):
        super().__init__()
        self.conv1 = nn.Conv2d(conv1_in, conv2_in, kernel_size=3, )
        self.conv2 = nn.Conv2d(conv2_in, conv3_in, 3)
        self.conv3 = nn.Conv2d(conv3_in, 24, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.dropout = nn.Dropout(p=.2)
        self.fc1 = nn.Linear(3456, first_dim)
        self.fc2 = nn.Linear(first_dim, second_dim)
        self.fc3 = nn.Linear(second_dim,third_dim)
        self.fc4 = nn.Linear(third_dim, _num_classes,)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        # fully connected layers 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x