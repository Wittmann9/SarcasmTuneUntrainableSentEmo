import torch.nn as nn
import torch.nn.functional as F

class FinalClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        return self.softmax(x)


class LinearTwoLayersNet(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim):
        super(LinearTwoLayersNet, self).__init__()
        self.linear_1 = nn.Linear(input_dim, middle_dim)
        self.activation = nn.ReLU()
        self.linear_11 = nn.Linear(middle_dim, middle_dim)
        self.linear_2 = nn.Linear(middle_dim, output_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_11(x)
        x = self.activation(x)
        return self.linear_2(x)
