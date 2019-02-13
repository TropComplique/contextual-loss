import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):

    def __init__(self, f):
        super(PerceptualLoss, self).__init__()
        self.f = nn.Parameter(data=f, requires_grad=False)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, c, h, w].
        Returns:
            a float tensor with shape [].
        """
        return F.mse_loss(x, self.f, reduction='mean')


class StyleLoss(nn.Module):

    def __init__(self, f):
        super(StyleLoss, self).__init__()
        self.g = nn.Parameter(data=gram_matrix(f), requires_grad=False)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [1, c, h, w].
        Returns:
            a float tensor with shape [].
        """
        return F.mse_loss(gram_matrix(x), self.g, reduction='mean')


def gram_matrix(x):
    """
    Arguments:
        x: a float tensor with shape [1, c, h, w].
    Returns:
        a float tensor with shape [c, c].
    """
    _, c, h, w = x.size()
    x = x.squeeze(0).view(c, h * w)
    g = torch.mm(x, x.t())
    return g.div(c * h * w)


class TotalVariationLoss(nn.Module):

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [].
        """
        h, w = x.size()[2:]
        # h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :(h - 1), :], 2)
        # w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :(w - 1)], 2)
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :(h - 1), :])
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :(w - 1)])
        return h_tv.mean([0, 1, 2, 3]) + w_tv.mean([0, 1, 2, 3])
