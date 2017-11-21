import torchvision.models as models
from torch import nn

OUTPUT_SIZE = 50


class ImageCNN(nn.Module):

    def __init__(self):
        super(ImageCNN, self).__init__()
        self.cnn = models.resnet152(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, OUTPUT_SIZE)
        self.bn = nn.BatchNorm1d(OUTPUT_SIZE, momentum=0.01)

    def forward(self, x):
        features = self.cnn(x)
        output = self.bn(features)
        return output
