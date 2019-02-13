import torch
import torch.nn as nn
import torch.nn.functional as F


# a small value
EPSILON = 1e-10


class ContextualLoss(nn.Module):
    """
    This computes CX(X, Y) where
    X and Y are sets of features.

    X = {x_1, ..., x_n} and Y = {y_1, ..., y_m},
    where x_i and y_j are spatial patches
    of features with shape [channels, size, size].

    It is assumed that Y is fixed and doesn't change!
    """

    def __init__(self, y, size, stride, h, distance='cosine'):
        """
        Arguments:
            y: a float tensor with shape [1, C, A, B].
            size, stride: integers, parameters of used patches.
            h: a float number.
            distance: a string, possible values are ['cosine', 'l2_squared'].
        """
        super(ContextualLoss, self).__init__()

        assert distance in ['cosine', 'l2_squared']
        normalize = distance == 'cosine'
        self.distance = distance

        y = y.squeeze(0)
        y_mu = torch.mean(y, dim=[1, 2], keepdim=True)  # shape [C, 1, 1]
        y = extract_patches(y - y_mu, size, stride, normalize)  # shape [M, C, size, size]

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

        if self.distance == 'cosine':
            similarity = cosine_similarity(x, self.y, self.stride)
            d = torch.clamp(1.0 - similarity, min=0.0, max=2.0)
        else:
            d = squared_l2_distance(x, self.y, self.stride)
            d = torch.clamp(d, min=0.0)

        # d has shape [N, M],
        # where N is the number of features on x
        # and M is the number of features on y

        d_min, _ = torch.min(d, dim=1, keepdim=True)
        # it has shape [N, 1]

        epsilon_from_the_paper = 1e-5
        d_tilde = d / (d_min + epsilon_from_the_paper)
        # it has shape [N, M]

        w = torch.exp(-d_tilde/self.h)  # shape [N, M]
        cx_ij = w / (torch.sum(w, dim=1, keepdim=True) + EPSILON)
        # it has shape [N, M]

        max_i_cx_ij, _ = torch.max(cx_ij, dim=0)  # shape [M]
        cx = torch.mean(max_i_cx_ij, dim=0)  # shape []
        cx_loss = -torch.log(cx + EPSILON)
        return cx_loss


def extract_patches(features, size, stride, normalize):
    """
    Arguments:
        features: a float tensor with shape [C, H, W].
        size: an integer, size of the patch.
        stride: an integer.
        normalize: a boolean.
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
    if normalize:
        norms = patches.view(M, -1).norm(p=2, dim=1)  # shape [M]
        patches /= (norms.view(M, 1, 1, 1) + EPSILON)

    return patches


def cosine_similarity(x, patches, stride):
    """
    Arguments:
        x: a float tensor with shape [C, H, W].
        patches: a float tensor with shape [M, C, size, size], normalized.
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


def squared_l2_distance(x, patches, stride):
    """
    Arguments:
        x: a float tensor with shape [C, H, W].
        patches: a float tensor with shape [M, C, size, size], unnormalized.
        stride: an integer.
    Returns:
        a float tensor with shape [N, M],
        where N = n * m, n = 1 + floor((H - size)/stride),
        and m = 1 + floor((W - size)/stride).
    """

    # compute squared norms of patches
    M = patches.size(0)
    patch_norms = torch.pow(patches, 2).sum(dim=[1, 2, 3])  # shape [M]

    # compute scalar products
    x = x.unsqueeze(0)
    products = F.conv2d(x, patches, stride=stride)  # shape [1, M, n, m]
    n, m = products.size()[2:]
    N = n * m
    products = products.squeeze(0).view(M, N)

    # compute squared norms of patches from x
    size = patches.size(2)
    x_norms = F.lp_pool2d(x, norm_type=2, kernel_size=size, stride=stride)  # shape [1, C, n, m]
    x_norms = torch.pow(x_norms, 2).sum(dim=1).squeeze(0).view(N)  # shape [N]

    # |x - y|^2 = |x|^2 + |y|^2 - 2*(x, y)
    distances = patch_norms.unsqueeze(1) + x_norms.unsqueeze(0) - 2.0 * products  # shape [M, N]
    return distances.t()
