import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.LayerNorm(output_dim)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        out += residual
        return out

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.LayerNorm(128)
        self.activation1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.3)
        
        self.res_block1 = ResidualBlock(128, 128)
        self.res_block2 = ResidualBlock(128, 128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.LayerNorm(64)
        self.activation2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = self.activation1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = self.activation2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x