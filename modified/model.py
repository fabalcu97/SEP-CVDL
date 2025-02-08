import torch.nn as nn
import torch.nn.functional as F

from variables import hyperparameters


class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(
            512, hyperparameters["conv5_filters"], kernel_size=3, padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(hyperparameters["conv5_filters"])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(
            hyperparameters["fc1_input"], hyperparameters["fc1_output"]
        )
        self.fc2 = nn.Linear(
            hyperparameters["fc1_output"], hyperparameters["fc2_output"]
        )
        self.dropout1 = nn.Dropout(hyperparameters["dropout1"])
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hyperparameters["fc2_output"], 6)

    def forward(self, x):  # (batch_size, channels=3, 64, 64)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, 64, 64)
        x = F.max_pool2d(x, 2)  # (batch_size, 64, 32, 32)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 128, 32, 32)
        x = F.max_pool2d(x, 2)  # (batch_size, 128, 16, 16)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch_size, 256, 16, 16)
        x = F.max_pool2d(x, 2)  # (batch_size, 256, 8, 8)
        x = self.dropout1(x)
        x = F.relu(self.bn4(self.conv4(x)))  # (batch_size, 512, 8, 8)
        x = F.max_pool2d(x, 2)  # (batch_size, 512, 4, 4)
        x = self.dropout1(x)
        x = F.relu(self.bn5(self.conv5(x)))  # (batch_size, 1024, 4, 4)
        x = F.max_pool2d(x, 2)  # (batch_size, 1024, 2, 2)
        # x = self.dropout1(x)

        x = self.pool(x)  # (batch_size, 1024, 1, 1)
        x = x.view(x.size(0), -1)  # (batch_size, 1024) # Flatten
        x = F.relu(self.fc1(x))  # (batch_size, 2048)
        x = self.dropout2(x)  # (batch_size, 2048)
        x = F.relu(self.fc2(x))  # (batch_size, 1024)
        x = self.fc3(x)  # (batch_size, 6)
        return x
