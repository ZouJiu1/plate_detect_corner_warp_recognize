import torch
import torch.nn as nn

# class Corner_net(nn.Module):
#     def __init__(self):
#         super(Corner_net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
#                                bias=False)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
#                                bias=False)
#         self.fc1 = nn.Linear(192, 128)#32 96  64 192
#         self.dropout = nn.Dropout(0.8)
#         self.fc2 = nn.Linear(128, 8)
#         self.sigmod = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.sigmod(x)
#         return x

class Corner_net(nn.Module):
    def __init__(self):
        super(Corner_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.fc1 = nn.Linear(96, 128)#32 96  64 192
        self.dropout = nn.Dropout(0.8)
        self.fc2 = nn.Linear(128, 8)
        self.sigmod = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmod(x)
        return x

def cornernet():
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model = Corner_net()
    return model
