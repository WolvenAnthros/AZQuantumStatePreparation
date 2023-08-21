import torch
import numpy as np

print(f'PyTorch ver: {torch.__version__}')

import torch.nn as nn
import torch.nn.functional as F

from TicTacToe import TicTacToe

torch.manual_seed(0)

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden,device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.startblock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backbone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32 * game.row_count * game.column_count, out_features=game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        x = self.startblock(x)
        for resBlock in self.backbone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


if __name__ == '__main__':
    tictactoe = TicTacToe()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = tictactoe.get_initial_state()
    state = tictactoe.get_next_state(state, 2, -1)
    state = tictactoe.get_next_state(state, 4, -1)
    state = tictactoe.get_next_state(state, 8, 1)
    state = tictactoe.get_next_state(state, 6, 1)

    print(state)
    encoded_state = tictactoe.get_encoded_state(state)

    print(f'\nencoded_state: {encoded_state}')

    tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0) # add additional brackets

    model = ResNet(tictactoe, 4,64, device=device)
    model.load_state_dict(torch.load('model_5.pt', map_location=device))
    model.eval()

    policy, value = model(tensor_state)
    value = value.item()
    policy = F.softmax(policy, dim=1).squeeze(0).cpu().detach().numpy()

    print(value, policy)