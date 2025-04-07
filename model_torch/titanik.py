import torch.nn as nn

class TitanicMLP(nn.Module):
    def __init__(self):
        super(TitanicMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class TitanicSLP(nn.Module):
    def __init__(self):
        super(TitanicSLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)