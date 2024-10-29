import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 32)
        self.fc7 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1)) + x1
        x3 = torch.relu(self.fc3(x2))
        x4 = torch.relu(self.fc4(x3)) + x3
        x5 = torch.relu(self.fc5(x4))
        x6 = torch.relu(self.fc6(x5)) + x5
        x = self.fc7(x6)
        return x