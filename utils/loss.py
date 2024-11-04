import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
BCE = torch.nn.BCELoss()

def keypoints_to_box(points):
    """
    通过4个关键点真值得到一个不旋转的矩形框。
    Args:
        keypoints: 关键点信息，包含4个点的坐标，形状为(4, 2)
    Returns:
        box: 表示矩形框的四个坐标，形如(x1, y1, x2, y2)
    """
    box = []
    for index,point in enumerate(points):
        x_max = int(max(point[:, 0]))
        x_min = int(min(point[:, 0]))
        y_max = int(max(point[:, 1]))
        y_min = int(min(point[:, 1]))
        box.append([x_min,y_min,x_max,y_max])
    return box

def iou(box1, box2):
    """
    计算两个矩形框的IoU。
    Args:
        box1: 矩形框1，形如(x1, y1, x2, y2)
        box2: 矩形框2，形如(x1, y1, x2, y2)
    Returns:
        iou: 两个矩形框的交并比（IoU），范围在[0, 1]之间，值越大表示重叠程度越高
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0

    return iou

def Iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    # 计算两个矩形框面积
    area1 = (xmax1-xmin1) * (ymax1-ymin1)
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1])) #计算交集面积
    iou = inter_area / (area1+area2-inter_area+1e-6) #计算交并比

    return iou

def IouLoss(box1,box2):
    # 计算预测框和目标框的左上角坐标和右下角坐标
    pred_boxe = np.array(box1)
    target_boxe = np.array(box2)
    iou_loss = 0
    batchSize = target_boxe.shape[0]
    for (bx1,bx2) in zip(pred_boxe,target_boxe):
        iou_loss +=1 - Iou(bx1,bx2)
    iou_loss = iou_loss/batchSize
    return iou_loss


# naive correlation板帧特征在搜索帧特征上滑动，
# 逐通道之间互相作内积，最后输出的就是一个通道数为1的特征
def naive_xcorr(self, z, x):
        # naive cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out

# depth-wise correlation思路上和naive correlation差不多，
# 只不过一个是全部通道加和起来了，一个就是一个通道就输出一个通道
def depthwise_xcorr(search, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    search = search.view(1, batch * channel, search.size(2), search.size(3))
    kernel = kernel.reshape(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = torch.nn.functional.conv2d(search, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

# 就是让模板特征的HzWz个1x1xC特征与搜索帧特征进行卷积，
# 最后的通道数是HzWz，大小因为核大小是1x1的，所以不会改变，就是Hx和Wx
def pixelwise_xcorr(kernel, search):
    b, c, h, w = search.shape
    ker = kernel.reshape(b, c, -1).transpose(1, 2)
    feat = search.reshape(b, c, -1)
    corr = torch.matmul(ker, feat)
    corr = corr.reshape(*corr.shape[:2], h, w)
    return corr


# pixel-to-global correlation
def pg_xcorr(kernel, search):
    b, c, h, w = search.shape
    ker1 = kernel.reshape(b, c, -1)
    ker2 = ker1.transpose(1, 2)
    feat = search.reshape(b, c, -1)
    S1 = torch.matmul(ker2, feat)
    S2 = torch.matmul(ker1, S1)
    corr = S2.reshape(*S2.shape[:2], h, w)
    return corr



def Dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1 - d

# class DiceLoss(nn.Module):
#     def __init__(self, num_classes=1) -> None:
#         super().__init__()
#         self.num_classes = num_classes
#         self.smooth = 1.

#     def forward(self, pred, target):
#         target = target.squeeze(1)
#         if self.num_classes == 1:
#             pred = torch.sigmoid(pred)
#             pred = pred.view(-1)
#             target = target.view(-1)
#             intersection = (pred * target).sum()
#             union = pred.sum() + target.sum()
#             loss = 1 -  (2. * intersection + self.smooth) / (union + self.smooth)
#         else:
#             pred = F.softmax(pred, dim=1)
#             loss = 0
#             for c in range(self.num_classes):
#                 pred_c = pred[:, c, :, :]
#                 target_c = (target == c).float()
#                 intersection = (pred_c * target_c).sum()
#                 union = pred_c.sum() + target_c.sum()
#                 loss += 1 -  (2. * intersection + self.smooth) / (union + self.smooth)
#             loss /= self.num_classes
#         return loss
    
class MLHingeLoss(nn.Module):

    def forward(self, x, y, reduction='mean'):
        """
            y: labels have standard {0,1} form and will be converted to indices
        """
        b, c = x.size()
        idx = (torch.arange(c) + 1).type_as(x)
        y_idx, _ = (idx * y).sort(-1, descending=True)
        y_idx = (y_idx - 1).long()

        return F.multilabel_margin_loss(x, y_idx, reduction=reduction)

def get_criterion(loss_name, **kwargs):

    losses = {
            "SoftMargin": nn.MultiLabelSoftMarginLoss,
            "Hinge": MLHingeLoss
            }

    return losses[loss_name](**kwargs)


#
# Mask self-supervision
#
def mask_loss_ce(mask, pseudo_gt, ignore_index=255):
    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    weight = pseudo_gt.sum(1).type_as(mask_gt)
    mask_gt += (1 - weight) * ignore_index

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index)
    return loss.mean()

def dice_coef(output, target):
    smooth = 1e-5

    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def weighted_loss(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()



def calc_loss(pred, target, bce_weight=0.5):
    bce = weighted_loss(pred, target)
    # dl = 1 - dice_coef(pred, target)
    # loss = bce * bce_weight + dl * bce_weight

    return bce


def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    loss1 = calc_loss(logit_S1, labels_S1)
    loss2 = calc_loss(logit_S2, labels_S2)

    return loss1 + loss2



def loss_diff(u_prediction_1, u_prediction_2, batch_size):
    a = weighted_loss(u_prediction_1, Variable(u_prediction_2, requires_grad=False))
    a = a.item()

    b = weighted_loss(u_prediction_2, Variable(u_prediction_1, requires_grad=False))
    b = b.item()

    loss_diff_avg = (a + b)
    return loss_diff_avg / batch_size


# def dice_loss(score, target):
#     target = target.float()
#     smooth = 1e-5
#     intersect = torch.sum(score * target)
#     y_sum = torch.sum(target * target)
#     z_sum = torch.sum(score * score)
#     loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#     loss = 1 - loss
#     return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class pDLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super(pDLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * ignore_mask)
        y_sum = torch.sum(target * target * ignore_mask)
        z_sum = torch.sum(score * score * ignore_mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        ignore_mask = torch.ones_like(target)
        ignore_mask[target == self.ignore_index] = 0
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


class SizeLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(SizeLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        output_counts = torch.sum(torch.softmax(output, dim=1), dim=(2, 3))
        target_counts = torch.zeros_like(output_counts)
        for b in range(0, target.shape[0]):
            elements, counts = torch.unique(
                target[b, :, :, :, :], sorted=True, return_counts=True)
            assert torch.numel(target[b, :, :, :, :]) == torch.sum(counts)
            target_counts[b, :] = counts

        lower_bound = target_counts * (1 - self.margin)
        upper_bound = target_counts * (1 + self.margin)
        too_small = output_counts < lower_bound
        too_big = output_counts > upper_bound
        penalty_small = (output_counts - lower_bound) ** 2
        penalty_big = (output_counts - upper_bound) ** 2
        # do not consider background(i.e. channel 0)
        res = too_small.float()[:, 1:] * penalty_small[:, 1:] + \
            too_big.float()[:, 1:] * penalty_big[:, 1:]
        loss = res / (output.shape[2] * output.shape[3] * output.shape[4])
        return loss.mean()


class MumfordShah_Loss(nn.Module):
    def levelsetLoss(self, output, target, penalty='l1'):
        # input size = batch x 1 (channel) x height x width
        outshape = output.shape
        tarshape = target.shape
        self.penalty = penalty
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:, ich], 1)
            target_ = target_.expand(
                tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2, 3)
                                  ) / torch.sum(output, (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - \
                pcentroid.expand(
                    tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss

    def gradientLoss2d(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if self.penalty == "l2":
            dH = dH * dH
            dW = dW * dW

        loss = torch.sum(dH) + torch.sum(dW)
        return loss

    def forward(self, image, prediction):
        loss_level = self.levelsetLoss(image, prediction)
        loss_tv = self.gradientLoss2d(prediction)
        return loss_level + loss_tv

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=19, ignore_index=19):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

        print("WARNING: SCE runs only with IGNORE INDEX = 19")

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes+1).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot[:,:,:,:self.num_classes].permute(0,3,1,2)), dim=1))
        #rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


# import torch
# from torch import nn
# import torch.nn.functional as F
# import numpy as np


# see https://github.com/charlesCXK/TorchSemiSeg/blob/main/furnace/seg_opr/loss_opr.py
class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index, reduction='mean', thresh=0.7, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

def compute_mean_intensity_inside_outside(pred, img):
    """
    Compute the mean intensity inside and outside the predicted segmentation.
    
    Args:
    - pred (torch.Tensor): Predicted segmentation map, sigmoid activated.
    - img (torch.Tensor): Corresponding image tensor.
    
    Returns:
    - Tuple (c1, c2): Mean intensity inside and outside the segmentation.
    """
    pred = torch.sigmoid(pred)  # Ensuring pred is in [0, 1] as a probability map
    img_mean_channel = img.mean(1, keepdim=True)  # Assuming img has shape (b, 3, h, w), reduce to (b, 1, h, w)

    # Mean intensity inside segmentation
    c1 = (pred * img_mean_channel).sum() / pred.sum()

    # Mean intensity outside segmentation
    c2 = ((1 - pred) * img_mean_channel).sum() / (1 - pred).sum()

    return c1, c2

def Energyloss(pred, img, lam1=1.0, lam2=1.0):
    """
    Compute the loss function based on predicted segmentation, image intensity, and dynamic mean intensities inside and outside the segmentation.
    
    Args:
    - pred (torch.Tensor): The predicted segmentation map with shape (b, 1, 256, 256).
    - img (torch.Tensor): The input image with shape (b, 3, 256, 256).
    - lam1, lam2 (float): Lambda coefficients for internal and external energies.
    
    Returns:
    - torch.Tensor: The computed loss value.
    """
    pred = torch.sigmoid(pred)  # Ensuring pred is in [0, 1] as a probability map

    # Calculate dynamic mean intensities
    c1, c2 = compute_mean_intensity_inside_outside(pred, img)

    # Calculate gradients of pred
    # grad_x, grad_y = torch.autograd.functional.jacobian(pred)
    # grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # Regularize to avoid zero division

    # Compute internal and external energy
    internal_energy = lam1 * ((img - c1) ** 2) * pred
    external_energy = lam2 * ((img - c2) ** 2) * (1 - pred)

    # Energy weighted by the gradient magnitude (considering boundary deformations)
    # L_ls = (internal_energy + external_energy) * grad_magnitude
    L_ls = (internal_energy + external_energy)


    # Mean over batch and spatial dimensions
    L_ls = L_ls.mean()

    return L_ls