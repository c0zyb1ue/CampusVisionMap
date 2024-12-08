
from torchvision import models
import torch
import torch.nn as nn


# Model
class Custom_ResNet(nn.Module) :
  def __init__(self):
    super().__init__()

    self.resnet = models.resnet18(pretrained=True)

    for p in self.resnet.parameters():
      p.requires_grad = False

    # add layer
    self.resnet.fc = torch.nn.Identity()

    self.fc_mid = nn.Linear(512, 256)  # New fully connected layer

    self.resnet.fc_building = nn.Linear(256, 30)


  def forward(self, x):
    out = self.resnet(x)
    out = torch.relu(self.fc_mid(out))
    building_class = self.resnet.fc_building(out)

    return building_class
