import torch
import torch.nn as nn
from pytorch_toolbelt.losses.functional import sigmoid_focal_loss, wing_loss
from torch.nn.modules.loss import MSELoss, SmoothL1Loss, _Loss
import torch.nn.functional as F


class LSRCrossEntropyLossV1(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', lb_ignore=-100):
        super(LSRCrossEntropyLossV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss



class LSRCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, label, lb_smooth, reduction, lb_ignore):
        # prepare label
        num_classes = logits.size(1)
        label = label.clone().detach()
        ignore = label == lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_pos, lb_neg = 1. - lb_smooth, lb_smooth / num_classes
        label = torch.empty_like(logits).fill_(
            lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(label.size(1)), *b]
        label[mask] = 0

        coeff = (num_classes - 1) * lb_neg + lb_pos
        ctx.coeff = coeff
        ctx.mask = mask
        ctx.logits = logits
        ctx.label = label
        ctx.reduction = reduction
        ctx.n_valid = n_valid

        loss = torch.log_softmax(logits, dim=1).neg_().mul_(label).sum(dim=1)
        if reduction == 'mean':
            loss = loss.sum().div_(n_valid)
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        coeff = ctx.coeff
        mask = ctx.mask
        logits = ctx.logits
        label = ctx.label
        reduction = ctx.reduction
        n_valid = ctx.n_valid

        scores = torch.softmax(logits, dim=1).mul_(coeff)
        scores[mask] = 0
        if reduction == 'none':
            grad = scores.sub_(label).mul_(grad_output.unsqueeze(1))
        elif reduction == 'sum':
            grad = scores.sub_(label).mul_(grad_output)
        elif reduction == 'mean':
            grad = scores.sub_(label).mul_(grad_output.div_(n_valid))
        return grad, None, None, None, None, None


class LSRCrossEntropyLossV2(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', lb_ignore=-100):
        super(LSRCrossEntropyLossV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = lb_ignore

    def forward(self, logits, label):
        return LSRCrossEntropyFunction.apply(
                logits, label, self.lb_smooth, self.reduction, self.lb_ignore)


def quad_kappa_loss_v2(predictions, labels, y_pow=2, eps=1e-9):
    # with tf.name_scope(name):
    #     labels = tf.to_float(labels)
    #     repeat_op = tf.to_float(
    #         tf.tile(tf.reshape(tf.range(0, num_ratings), [num_ratings, 1]), [1, num_ratings]))
    #     repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
    #     weights = repeat_op_sq / tf.to_float((num_ratings - 1) ** 2)

    batch_size = predictions.size(0)
    num_ratings = predictions.size(1)
    assert predictions.size(1) == num_ratings

    tmp = torch.arange(0, num_ratings).view((num_ratings, 1)).expand((-1, num_ratings)).float()
    weights = (tmp - torch.transpose(tmp, 0, 1)) ** 2 / (num_ratings - 1) ** 2
    weights = weights.type(labels.dtype).to(labels.device)

    pred_ = predictions ** y_pow
    pred_norm = pred_ / (eps + torch.sum(pred_, 1).view(-1, 1))

    hist_rater_a = torch.sum(pred_norm, 0)
    hist_rater_b = torch.sum(labels, 0)

    conf_mat = torch.matmul(pred_norm.t(), labels)

    nom = torch.sum(weights * conf_mat)
    denom = torch.sum(
        weights * torch.matmul(hist_rater_a.view(num_ratings, 1), hist_rater_b.view(1, num_ratings)) / batch_size)
    return -(1.0 - nom / (denom + eps))


class HybridCappaLoss(nn.Module):
    # TODO: Test
    # https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master/losses.py#L51
    def __init__(self, y_pow=2, log_scale=1.0, eps=1e-15, log_cutoff=0.9, ignore_index=None, gamma=2.):
        super().__init__()
        self.y_pow = y_pow
        self.log_scale = log_scale
        self.log_cutoff = log_cutoff
        self.eps = eps
        self.ignore_index = ignore_index
        self.gamma = 2

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0.).to(input.device)

        focal_loss = 0
        num_classes = input.size(1)
        for cls in range(num_classes):
            cls_label_target = (target == cls).long()
            cls_label_input = input[:, cls]
            focal_loss += sigmoid_focal_loss(cls_label_input, cls_label_target, gamma=self.gamma, alpha=None)

        # Second term
        y = F.log_softmax(input, dim=1).exp()
        target_one_hot = F.one_hot(target, input.size(1)).float()
        # +1 to make loss be [0;2], instead [-1;1]
        kappa_loss = 1 + quad_kappa_loss_v2(y, target_one_hot, y_pow=self.y_pow, eps=self.eps)

        return kappa_loss + self.log_scale * focal_loss
