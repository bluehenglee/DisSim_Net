import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
from PIL import Image

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


def get_iou(mask_name,predict):
    image_mask = cv2.imread(mask_name,0)
    image_mask = cv2.resize(image_mask, (256, 256))
    # mask_org = Image.fromarray(image_mask)
    # image_mask = np.array(mask_org.resize((512, 512), Image.BICUBIC))
    # image_mask = cv2.resize(image_mask,dsize=(256,256),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    # image_mask = mask_name
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(256,256))
    # image_mask = mask
    # print(image.shape)
    height = predict.shape[0]
    weight = predict.shape[1]
    # print(height*weight)
    o = 0
    for row in range(height):
            for col in range(weight):
                if predict[row, col] < 0.5:  #由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                    predict[row, col] = 0
                else:
                    predict[row, col] = 1
                if predict[row, col] == 0 or predict[row, col] == 1:
                    o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
            for col in range(weight_mask):
                if image_mask[row, col] < 125:   #由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
                    image_mask[row, col] = 0
                else:
                    image_mask[row, col] = 1
                if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                    o += 1
    predict = predict.astype(np.int16)

    interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    iou_tem = (inter+0.0000000001) / (union+0.0000000001)

    # Iou = IOUMetric(2)  #2表示类别，肝脏类+背景类
    # Iou.add_batch(predict, image_mask)
    # a, b, c, d, e= Iou.evaluate()
    # print('%s:iou=%f' % (mask_name,iou_tem))

    return iou_tem

def get_dice(mask_name,predict):
    image_mask = cv2.imread(mask_name, 0)
    image_mask = cv2.resize(image_mask, (256, 256))

    # mask_org = Image.fromarray(image_mask)
    # image_mask = np.array(mask_org.resize((512, 512), Image.BICUBIC))
    # image_mask = cv2.resize(image_mask,dsize=(256,256),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    # image_mask = mask_name
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(256,256))
    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:  # 由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
            if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                o += 1
    predict = predict.astype(np.int16)
    intersection = (predict*image_mask).sum()
    dice = (2. *intersection+0.0000000001) /(predict.sum()+image_mask.sum()+0.0000000001)
    return dice

# def get_hd(mask_name,predict):
#     image_mask = cv2.imread(mask_name, 0)#flag = 0，8位深度，1通道
#     image_mask = cv2.resize(image_mask, (256, 256))
#     # mask_org = Image.fromarray(image_mask)
#     # image_mask = np.array(mask_org.resize((512, 512), Image.BICUBIC))
#     # image_mask = cv2.resize(image_mask,dsize=(256,256),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
#     # image_mask = mask_name
#     # print(mask_name)
#     # print(image_mask)
#     if np.all(image_mask == None):
#         image_mask = imageio.mimread(mask_name)
#         image_mask = np.array(image_mask)[0]
#         image_mask = cv2.resize(image_mask,(256,256))

#     #image_mask = mask
#     height = predict.shape[0]
#     weight = predict.shape[1]
#     o = 0
#     for row in range(height):
#         for col in range(weight):
#             if predict[row, col] < 0.5:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
#                 predict[row, col] = 0
#             else:
#                 predict[row, col] = 1
#             if predict[row, col] == 0 or predict[row, col] == 1:
#                 o += 1
#     height_mask = image_mask.shape[0]
#     weight_mask = image_mask.shape[1]
#     for row in range(height_mask):
#         for col in range(weight_mask):
#             if image_mask[row, col] < 125:  # 由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
#                 image_mask[row, col] = 0
#             else:
#                 image_mask[row, col] = 1
#             if image_mask[row, col] == 0 or image_mask[row, col] == 1:
#                 o += 1
#     # print(image_mask.shape,predict.shape)
#     hd1 = directed_hausdorff(image_mask, predict)[0]
#     hd2 = directed_hausdorff(predict, image_mask)[0]
#     res = None
#     if hd1>hd2 or hd1 == hd2:
#         res=hd1
#         return res
#     else:
#         res=hd2
#         return res

def get_hd(true_mask, predict):
    # Assuming both true_mask and predict are PyTorch tensors on CPU and need to be moved to GPU.
    # Binarize the predictions: anything above or equal to 0.5 is considered as 1, otherwise 0.
    predict = (predict >= 0.5).float()

    # Calculate the directed Hausdorff distance using the CPU as scipy does not support GPU.
    true_mask_np = true_mask.cpu().numpy()
    predict_np = predict.cpu().numpy()

    # Compute Hausdorff distances.
    hd1 = directed_hausdorff(true_mask_np, predict_np)[0]
    hd2 = directed_hausdorff(predict_np, true_mask_np)[0]

    # Return the maximum of the two distances.
    return max(hd1, hd2)

def get_precision(mask_name, predict):
    image_mask = cv2.imread(mask_name, 0)
    image_mask = cv2.resize(image_mask, (256, 256))

    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask, (256, 256))

    height = predict.shape[0]
    weight = predict.shape[1]

    # Binarize predict array
    predict = np.where(predict < 0.5, 0, 1)

    # Binarize image_mask array
    image_mask = np.where(image_mask < 125, 0, 1)

    predict = predict.astype(np.int16)
    image_mask = image_mask.astype(np.int16)

    # Calculate True Positives (TP) and False Positives (FP)
    TP = np.sum((predict == 1) & (image_mask == 1))
    FP = np.sum((predict == 1) & (image_mask == 0))

    # Calculate Precision
    precision = TP / (TP + FP + 1e-10)  # Adding a small epsilon to avoid division by zero
    return precision

def show(predict):
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            predict[row, col] *= 255
    plt.imshow(predict)
    plt.show()