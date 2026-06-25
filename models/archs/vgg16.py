import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Vgg16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def load(self, model_folder='vgg16'):
        """load the vgg16 model feature"""
        if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
            if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
                os.system(  # download vgg16.t7
                    'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(model_folder, 'vgg16.t7')
                )
            vgglua = torch.load(os.path.join(model_folder, 'vgg16.t7'))
            vgg = Vgg16()
            for src, dst in zip(vgglua.parameters()[0], vgg.parameters()):
                dst.data[:] = src
            torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))
        self.load_state_dict(torch.load(os.path.join(model_folder, 'vgg16.weight')))

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        return relu3_3
