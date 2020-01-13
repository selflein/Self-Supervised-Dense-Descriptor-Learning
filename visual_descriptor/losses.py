import math

import torch
from torch import nn
import torch.nn.functional as F


def feature_distance(feat1, feat2, eps=1e-7, sqrt=True):
    """Compute L2 distance between feature vectors.

    Args:
        feat1: Tensor of shape (b, num_points, num_features).
        feat2: Same as feat1.
    Returns:
        Tensor of shape (b, num_points) with losses between feature
        vectors.
    """
    diff = torch.pow((feat1 - feat2), 2).sum(-1)
    if sqrt:
        diff = (diff + eps).sqrt()
    return diff


def features_from_image(img_feat, pos):
    """Extract points from image.

    Args:
        img_feat: Batch image of shape (b, c, w, h).
        pos: Indices into the batch image of shape (b, n, 2) where n is the
            number of points.
    Returns:
        Image points at requested indices of shape (b, n, c).
    """
    b = img_feat.size(0)
    return img_feat[torch.arange(b)[:, None], :, pos[:, :, 0], pos[:, :, 1]]


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, neg_loss_weight=1., sim_hard_negative=True):
        super().__init__()
        self.margin = margin
        self.neg_loss_weight = neg_loss_weight
        self.pos_criterion = PositiveLoss(margin)
        self.neg_criterion = NegativeLoss(margin, sim_hard_negative)

    def forward(self, out_1, out_2, match_1, match_2, nonmatch_2):
        """
        Args:
            out_1: Tensor of shape (b, c, w, h)
            out_2: Tensor of shape (b, c, w, h)
            match_1: Tensor of shape (b, n, 2) with matching coordinates in
                first image.
            match_2: Tensor of shape (b, n, 2) with matching coordinates in
                second image.
            nonmatch_2: Tensor of shape (b, m, n, 2) with m non-matching points
                in second image.
        """
        """ Matching points """
        pos_loss = self.pos_criterion(out_1, out_2, match_1, match_2, nonmatch_2)

        """ Non matching points """
        neg_loss = self.neg_criterion(out_1, out_2, match_1, match_2, nonmatch_2)
        return pos_loss + self.neg_loss_weight * neg_loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, out_1, out_2, xy_1, xy_2, nonmatch_2):
        """
        Args:
            out_1: Tensor of shape (b, c, w, h)
            out_2: Tensor of shape (b, c, w, h)
            xy_1: Tensor of shape (b, n, 2)
            xy_2: Tensor of shape (b, n, 2)
            nonmatch_2: Tensor of shape (b, m, n, 2). m negative points for every
                positive point.
        """
        """ Matching points """
        out_1_match = features_from_image(out_1, xy_1)
        out_2_match = features_from_image(out_2, xy_2)
        pos_dists = feature_distance(out_1_match, out_2_match)

        """ Non matching points """
        b, m, n, _ = nonmatch_2.shape
        out_2_nonmatch = features_from_image(out_2, nonmatch_2.view(b, -1, 2))
        neg_dists = feature_distance(out_1_match.repeat(1, m, 1), out_2_nonmatch)
        neg_dists = neg_dists.view(b, m, n).mean(1)

        loss = F.relu(pos_dists - neg_dists + self.margin).mean()
        return loss


class PositiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, out_1, out_2, match_1, match_2, nonmatch_2):
        """
        Args:
            out_1: Tensor of shape (b, c, w, h)
            out_2: Tensor of shape (b, c, w, h)
            match_1: Tensor of shape (b, n, 2) with matching coordinates in
                first image.
            match_2: Tensor of shape (b, n, 2) with matching coordinates in
                second image.
            nonmatch_2: Tensor of shape (b, n, 2) with non-matching points
                in second image.
        """
        """ Matching points """
        out_1_match = features_from_image(out_1, match_1)
        out_2_match = features_from_image(out_2, match_2)
        pos_loss = feature_distance(out_1_match, out_2_match, sqrt=False).mean()

        return pos_loss


class NegativeLoss(nn.Module):
    def __init__(self, margin=0.5, sim_hard_negative=True):
        super().__init__()
        self.sim_hard_negative = sim_hard_negative
        self.margin = margin

    def forward(self, out_1, out_2, match_1, match_2, nonmatch_2):
        """
        Args:
            out_1: Tensor of shape (b, c, w, h)
            out_2: Tensor of shape (b, c, w, h)
            match_1: Tensor of shape (b, n, 2)
            match_2: Tensor of shape (b, n, 2)
            nonmatch_2: Tensor of shape (b, m, n, 2). m negative points for every
                positive point.
        """
        """ Points in first imagge """
        out_1_match = features_from_image(out_1, match_1)

        """ Non matching points """
        b, m, n, _ = nonmatch_2.shape
        out_2_nonmatch = features_from_image(out_2, nonmatch_2.view(b, -1, 2))
        neg_dists = feature_distance(out_1_match.repeat(1, m, 1), out_2_nonmatch, sqrt=False)
        if self.sim_hard_negative:
            diff = self.margin - neg_dists
            n_hard_negatives = (diff > 0.).float().sum()
            neg_loss = F.relu(diff).sum() / n_hard_negatives
        else:
            neg_loss = F.relu(self.margin - neg_dists).mean()

        return neg_loss


class PixelwiseContrastiveLoss(torch.nn.Module):
    """ Taken from https://gist.github.com/peteflorence/4c009e7dd5eee7b5c8caa2c9bae954d5#file-pixelwise_contrastive_loss-py-L10"""

    def __init__(self):
        super(PixelwiseContrastiveLoss, self).__init__()
        self.num_non_matches_per_match = 150

    def forward(self, image_a_pred, image_b_pred, matches_a, matches_b,
                non_matches_a, non_matches_b):
        """
        Computes the loss function
        DCN = Dense Correspondence Network
        num_images = number of images in this batch
        num_matches = number of matches
        num_non_matches = number of non-matches
        W = image width
        H = image height
        D = descriptor dimension
        match_loss = 1/num_matches \sum_{num_matches} ||descriptor_a - descriptor_b||_2^2
        non_match_loss = 1/num_non_matches \sum_{num_non_matches} max(0, M_margin - ||descriptor_a - descriptor_b||_2^2 )
        loss = match_loss + non_match_loss
        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_b
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :type non_matches_b:
        :return: loss, match_loss, non_match_loss
        :rtype: torch.Variable(torch.FloatTensor) each of shape torch.Size([1])
        """
        b, c, h, w = image_a_pred.shape
        assert b == 1, 'Loss works only with batch size 1'

        # Flatten image pixels
        image_a_pred = image_a_pred.view(b, c, h * w).transpose(1, 2)
        image_b_pred = image_b_pred.view(b, c, h * w).transpose(1, 2)

        # Remove batch dim
        matches_a = matches_a.squeeze(0)
        matches_b = matches_b.squeeze(0)
        non_matches_a = non_matches_a.squeeze(0)
        non_matches_b = non_matches_b.squeeze(0)

        # Compute indices
        matches_a = matches_a[:, 0] * w + matches_a[:, 1]
        matches_b = matches_b[:, 0] * w + matches_b[:, 1]
        non_matches_a = non_matches_a[:, 0] * w + non_matches_a[:, 1]
        non_matches_b = non_matches_b[:, 0] * w + non_matches_b[:, 1]

        loss = 0

        # add loss via matches
        matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)
        loss += (matches_a_descriptors - matches_b_descriptors).pow(2).sum(2).mean()
        match_loss = 1.0 * loss.item()

        # add loss via non_matches
        M_margin = 0.5  # margin parameter
        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)
        pixel_wise_loss = (non_matches_a_descriptors - non_matches_b_descriptors).pow(2).sum(dim=2)
        pixel_wise_loss = torch.add(torch.neg(pixel_wise_loss), M_margin)
        zeros_vec = torch.zeros_like(pixel_wise_loss)
        loss += torch.max(zeros_vec, pixel_wise_loss).mean()
        non_match_loss = loss.item() - match_loss

        return loss, match_loss, non_match_loss


if __name__ == '__main__':
    test = torch.arange(256 * 256 * 3).reshape(1, 3, 256, 256)
    idx = torch.arange(20 * 2).reshape(1, 20, 2)
    f = features_from_image(test, idx)

    out = []
    for i in idx[0]:
        out.append(test[:, :, i[0], i[1]])
    out = torch.stack(out, dim=1)
    print((f == out).all())

    loss_func_1 = PixelwiseContrastiveLoss()
    loss_func_2 = ContrastiveLoss()

    nonmatch_1 = torch.randint(0, 256, (1, 50, 2)).long()
    nonmatch_2 = torch.randint(0, 256, (1, 50, 2)).long()
    match_1 = torch.randint(0, 256, (1, 50, 2)).long()
    match_2 = torch.randint(0, 256, (1, 50, 2)).long()

    test_img1 = torch.randn((1, 3, 256, 256))
    test_img2 = torch.randn((1, 3, 256, 256))

    loss1 = loss_func_1(test_img1, test_img2, match_1, match_2,
                nonmatch_1, nonmatch_2)[0]
    loss2 = loss_func_2(test_img1, test_img2, match_1, match_2,
                nonmatch_1, nonmatch_2)

    print(loss1, loss2)

