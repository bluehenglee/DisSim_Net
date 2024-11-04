import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def extract_image_patches(img,ksize=3,padding=1,stride=1):
    # B,C,H,W = img.shape#b*1*h*w
    img = img.cuda()
    k_zeros = torch.zeros(ksize,ksize)
    kernel_list = []
    for i in range(ksize):
        for j in range(ksize):
            k = k_zeros.clone()
            k[i][j] = 1#有九种卷积核，每个3*3卷积核对应位置为1，其余8个位置为0
            kernel_list.append(k)#列表里有9个3*3卷积核
    kernel = torch.stack(kernel_list)#9*3*3
    kernel = kernel.unsqueeze(1).cuda()#9*1*3*3
    weight = nn.Parameter(data=kernel,requires_grad=False)
    out = F.conv2d(img,weight,padding=padding,stride=stride)#b*9*256*256

    return out
# img = torch.ones(8,1,256,256)
# img_patchs = extract_image_patches(img,ksize=3,padding=1,stride=1)
# print(img_patchs.shape)
# print(1e-9)


# x是原图
# kernel_size计算相似度的卷积核大小
# dilation空洞卷积的膨胀系数controls the spacing between the kernel points;
# 相当于计算了每一个位置的相对的8个临近的位置的像素，然后保存为一个8维数据
def unfold_wo_center(x, kernel_size, dilation):

    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2#两条/，得到整数部分，小数部分去掉，dilation为2，padding为2
    #  F.unfold具体输入是[batchsize, channel, h, w]
    # 输出是[batchsize, channel * kH * kW, L]其中kH是kernel的高，kW是kernel宽，L则是这个高kH宽kW的kernel能在H*W区域按照指定stride滑动的次数。
    # 相当于给每个像素卷了但是不积（不求卷积后的值，直接平铺数值）
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )#b*c*h*w-->b*(c*9)*(h*w)

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )#b*c*9*h*w

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)
    # 扩张一维，新增的一维为8，代表8个相邻点，后续用来计算相邻的点与中心点的相似度
    return unfolded_x#b*c*8*h*w


def get_images_color_similarity(images, image_masks, kernel_size, dilation):#传入原图和mask计算相似性
    assert images.dim() == 4
    # assert images.size(0) == 1

    # 原图8个临近点生成的8维的图片
    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )
    # images和unfold_images的相似性，即和他每个相邻点的相似性，一个维度代表某一个角落的点的合集
    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    # image_mask预测图包裹的地方8个临近点的weight图
    unfolded_weights = unfold_wo_center(
        image_masks, kernel_size=kernel_size,
        dilation=dilation
    )
    # 算8个相邻点中和前景最相似的作为weight
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights#b*8*h*w


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):#计算(ye=1)
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)#B*C*H*W
    log_bg_prob = F.logsigmoid(-mask_logits)#因为log1=0，所以这么写

    # 前景预测和背景预测与邻近的8个点之间预测
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )#B*C*8*H*W
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )#B*C*8*H*W

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the probability in log space to avoid numerical instability
    # 预测的前景像素向量和每个相邻像素向量相乘（两个点同时为前景的概率+两个点同时为背景的概率）ye为1
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold  #因为这里是log,所以*即是+
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold
    # [:, :, None]数据会增加一维

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)  #前景相似概率和背景相似概率都要计算，取最大只要相似就行
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]#b*8*h*w
