import torch
import torch.nn as nn
from torchvision.models import vgg19


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()

        self.model = vgg19(pretrained=True).eval().features
        for p in self.model.parameters():
            p.requires_grad = False

        # normalization
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.mean = nn.Parameter(data=mean, requires_grad=False)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.std = nn.Parameter(data=std, requires_grad=False)

        names = []
        i, j = 1, 1
        for m in self.model:

            if isinstance(m, nn.Conv2d):
                names.append(f'conv{i}_{j}')

            elif isinstance(m, nn.ReLU):
                names.append(f'relu{i}_{j}')
                m.inplace = False
                j += 1

            elif isinstance(m, nn.MaxPool2d):
                names.append(f'pool{i}')
                i += 1
                j = 1

        # feature names
        self.names = names

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents RGB images with pixel values in [0, 1] range.
        Returns:
            a dict with float tensors.
        """
        features = {}
        x = (x - self.mean)/self.std

        for n, m in zip(self.names, self.model):
            x = m(x)
            features[n] = x

        return features
