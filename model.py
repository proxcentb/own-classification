import torch.nn as nn

class Net(nn.Module):
    def __init__(self, outFeatures):
        super(Net, self).__init__()
        self.relu = nn.LeakyReLU()
        self.pool = nn.AvgPool2d(4, 4)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False, padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False, padding_mode='zeros')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=outFeatures, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros')

    def forward(self, x, train=True):
        x = self.pool(self.relu(self.conv1(x)))
        x = nn.functional.dropout2d(x, p=0.5, training=train)
        x = self.pool(self.relu(self.conv2(x)))
        x = nn.functional.dropout2d(x, p=0.5, training=train)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.softmax(x)
        return x


