import torch
import torch.nn as nn
import torch.nn.functional as F


# a small value
EPSILON = 1e-8


class ContextualLoss(nn.Module):
    """
    This computes CX(X, Y) where
    X and Y are sets of features.

    X = {x_1, ..., x_n} and Y = {y_1, ..., y_m},
    where x_i and y_j are spatial patches
    of features with shape [channels, size, size].

    It is assumed that Y is fixed and doesn't change!
    """

    def __init__(self, y, size, stride, h):
        """
        Arguments:
            y: a float tensor with shape [1, C, A, B].
            size, stride: integers, parameters of used patches.
            h: a float number.
        """
        super(ContextualLoss, self).__init__()

        y = y.squeeze(0)
        y_mu = torch.mean(y, dim=[1, 2], keepdim=True)  # shape [C, 1, 1]
        y = extract_patches(y - y_mu, size, stride)  # shape [M, C, size, size]

        self.y_mu = nn.Parameter(data=y_mu, requires_grad=False)
        self.y = nn.Parameter(data=y, requires_grad=False)

        self.stride = stride
        self.h = h

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [1, C, H, W].
        Returns:
            a float tensor with shape [].
        """
        x = x.squeeze(0)
        x = x - self.y_mu  # shape [C, H, W]

        similarity = cosine_similarity(x, self.y, self.stride)  # shape [N, M]
        # where N is the number of features on x
        # and M is the number of features on y

        d = 1.0 - similarity
        d_min, _ = torch.min(d, dim=1, keepdim=True)
        # it has shape [N, 1]

        epsilon_from_the_paper = 1e-5
        d_tilde = d / (d_min + epsilon_from_the_paper)
        # it has shape [N, M]

        w = torch.exp((1.0 - d_tilde) / self.h)  # shape [N, M]
        cx_ij = w / (torch.sum(w, dim=1, keepdim=True) + EPSILON)
        # it has shape [N, M]

        max_i_cx_ij, _ = torch.max(cx_ij, dim=0)  # shape [M]
        cx = torch.mean(max_i_cx_ij, dim=0)  # shape []
        cx_loss = -torch.log(cx + EPSILON)
        return cx_loss


def extract_patches(features, size, stride):
    """
    Arguments:
        features: a float tensor with shape [C, H, W].
        size: an integer, size of the patch.
        stride: an integer.
    Returns:
        a float tensor with shape [M, C, size, size],
        where M = n * m, n = 1 + floor((H - size)/stride),
        and m = 1 + floor((W - size)/stride).
    """
    C = features.size(0)
    patches = features.unfold(1, size, stride).unfold(2, size, stride)
    # it has shape [C, n, m, size, size]

    # get the number of patches
    n, m = patches.size()[1:3]
    M = n * m

    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(M, C, size, size)
    norms = patches.view(M, -1).norm(p=2, dim=1)  # shape [M]
    patches /= (norms.view(M, 1, 1, 1) + EPSILON)

    return patches


def cosine_similarity(x, patches, stride):
    """
    Arguments:
        x: a float tensor with shape [C, H, W].
        patches: a float tensor with shape [M, C, size, size].
        stride: an integer.
    Returns:
        a float tensor with shape [N, M],
        where N = n * m, n = 1 + floor((H - size)/stride),
        and m = 1 + floor((W - size)/stride).
    """
    M = patches.size(0)
    x = x.unsqueeze(0)
    products = F.conv2d(x, patches, stride=stride)  # shape [1, M, n, m]

    size = patches.size(2)
    x_norms = F.lp_pool2d(x, norm_type=2, kernel_size=size, stride=stride)  # shape [1, C, n, m]
    x_norms = x_norms.norm(p=2, dim=1, keepdim=True)  # shape [1, 1, n, m]
    products /= (x_norms + EPSILON)

    return products.squeeze(0).view(M, -1).t()
