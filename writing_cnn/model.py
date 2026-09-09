import torch
import torch.nn as nn


# Residual block

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0, se_reduction=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        hidden = max(out_channels // se_reduction, 4)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
            nn.Sigmoid()
        )

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = x * self.se(x)

        return self.relu(x + identity)


# Writing CNN

class WritingCNN(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, dropout=0.05),
            ResidualBlock(32, 32, dropout=0.05),
        )

        self.block2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, dropout=0.08),
            ResidualBlock(64, 64, dropout=0.08),
        )

        self.block3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout=0.10),
            ResidualBlock(128, 128, dropout=0.10),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, number_of_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        average = self.avg_pool(x).flatten(1)
        maximum = self.max_pool(x).flatten(1)
        x = torch.cat((average, maximum), dim=1)

        return self.classifier(x)