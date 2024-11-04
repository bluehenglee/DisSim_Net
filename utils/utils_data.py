import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from PIL import Image, ImageDraw
from scipy import ndimage

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def keypoints_to_box(points):
    """
    通过4个关键点真值得到一个不旋转的矩形框。
    Args:
        keypoints: 关键点信息，包含4个点的坐标，形状为(B，4, 2)
    Returns:
        box: 表示矩形框的四个坐标，形如(x1, y1, x2, y2)
    """
    pointss = np.array(points)
    box = []
    for index,point in enumerate(pointss):
        x_max = int(max(point[:, 0]))
        x_min = int(min(point[:, 0]))
        y_max = int(max(point[:, 1]))
        y_min = int(min(point[:, 1]))
        box.append([x_min,y_min,x_max,y_max])
    # print(box)
    return box

def keypoints_to_mask(points,x,mask_path):
    """
    通过4个关键点真值得到一个不旋转的矩形框。
    Args:
        keypoints: 关键点信息,包含4个点的坐标,形状为(B,4, 2)
    Returns:
        box: 表示矩形框的四个坐标，形如(x1, y1, x2, y2)
    """
    output_path = 'data/tn3k/train/kpmask'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    b,c,h,w = x.shape
    pointss = np.array(points)
    mask = torch.zeros(b,1,h,w).to(x.device)
    for index,point in enumerate(pointss):
        x_max = int(max(point[:, 0]))
        x_min = int(min(point[:, 0]))
        y_max = int(max(point[:, 1]))
        y_min = int(min(point[:, 1]))
        mask[index, :, y_min: y_max+1, x_min: x_max+1] = 1
        # mask[y_min: y_max+1, x_min: x_max+1] = 1
        # print(mask)
        plt.imsave(output_path + '/' + mask_path[0].split('/')[-1], mask[index].squeeze(), cmap='Greys_r')
    # n,c,h,w
    x_label = mask.max(dim=2,keepdim=True)[0]
    y_label = mask.max(dim=3, keepdim=True)[0]
    # print(box)
    return mask,x_label,y_label

def seg2box(masks):
    """
    通过分割结果得到一个不旋转的矩形框。
    Args:
        masks: 分割的mask，nparray形式，形状为(B，256，256）
    """
    mask = np.copy(masks)
    # 将分割结果转换为二进制图像，即mask
    # flag = [] # 根据具体分割算法阈值进行选择
    mask[masks > 0.5] = 1
    mask[masks <= 0.5] = 0


    # mask = cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # print(mask)

    box = []
    for m in mask:
        m = m.astype(np.uint8)
        # m = m.squeeze()
        # 使用findContours函数找到轮廓
        contours, hierarchy = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        # 轮廓点个数

        # 使用boundingRect函数计算边界框的坐标
        x, y, w, h = cv2.boundingRect(max_contour)
        x1, y1, x2, y2 = x, y, x+w, y+h
        box.append([x1,y1,x2,y2])
        num = max_contour.shape[0]
        # print(num)
    return box,num


def get_roi(points, pic):
    """
    通过4个关键点真值得到pic感兴趣区域的图像。
    Args:
        points: 关键点信息,包含4个点的坐标,形状为(B,4)  x1,y1,x2,y2
    Returns:
        pic: 待提取的图片,torch.tensor形式,形状为(B,C, H,W)
    """
    pointss = np.array(points)
    roi = torch.zeros_like(pic).to(pic.device)

    for index, point in enumerate(pointss):
        x_min = int(point[0])
        y_min = int(point[1])
        x_max = int(point[2])
        y_max = int(point[3])
        # x_max = min(int(max(point[:,0])) + 10, 255)
        # x_min = max(int(min(point[:,0])) - 10, 0)
        # y_max = min(int(max(point[:,1])) + 10, 255)
        # y_min = max(int(min(point[:,1])) - 10, 0)
        # kp.append([y_min, y_max, x_min, x_max])
        roi[index, :, y_min: y_max+1, x_min: x_max+1] = 1
        # roi[index, :, y_min: y_max+1, x_min: x_max+1] = 1
    # print(roi)
        # roi_kp.append(kp)
    return roi * pic


def get_union(pred_boxes, target_boxes, images):
    """
    计算带两个矩形框的并集区域
    :param pred_boxes: 预测的矩形框（x1, y1, x2, y2）
    :param target_boxes: 目标矩形框（x1, y1, x2, y2）
    :param images:待提取联合区域的图片，torch.tensor(B, C, H, W)
   """
    n,c,h,w = images.size()
    roi = torch.zeros_like(images)
    # 计算预测框和目标框的左上角坐标和右下角坐标
    pred_boxe = np.array(pred_boxes)
    target_boxe = np.array(target_boxes)
    # pred_x1中包含4个数,pred_y1都包含4个数
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxe[:, 0], pred_boxe[:, 1], pred_boxe[:, 2], pred_boxe[:, 3]
    target_x1, target_y1, target_x2, target_y2 = target_boxe[:, 0], target_boxe[:, 1], target_boxe[:, 2], target_boxe[:, 3]

    # 计算交集，4个数
    x1 = np.minimum(pred_x1, target_x1)
    y1 = np.minimum(pred_y1, target_y1)
    x2 = np.maximum(pred_x2, target_x2)
    y2 = np.maximum(pred_y2, target_y2)
    for i in range(n):
        roi[i,:,y1[i]:y2[i]+1,x1[i]:x2[i]+1] = images[i,:,y1[i]:y2[i]+1,x1[i]:x2[i]+1]

    # print(torch.max(roi))
    # 计算并集
    # union_area = [inter_x1,inter_y1,inter_x2,inter_y2]
    return roi

def point_to_fore(point,mask_path):
    # 创建一个空白的图像
    points = np.copy(point)
    points = point.reshape(-1,2)
    points[[1,2],:] = points[[2,1],:]
    x_max = int(max(points[:, 0]))
    x_min = int(min(points[:, 0]))
    y_max = int(max(points[:, 1]))
    y_min = int(min(points[:, 1]))
    # points = points.tolist()
    points = [tuple(x) for x in points.tolist()]

    mask = Image.new('L', (256, 256), 0)
    # 创建一个绘图对象
    draw = ImageDraw.Draw(mask)
    print(points[-1])
    # 将第一个点加入路径，并迭代处理剩余顶点
    draw.line((points[-1], points[0]), fill=255, width=1)
    for i in range(len(points)-1):
        draw.line((points[i], points[i+1]), fill=255, width=1)

    # 使用序列洪泛算法将内部填充为255
    x = (x_min + x_max)/2
    y = (y_min + y_max)/2
    ImageDraw.floodfill(mask, (x, y), 255)
    mask = mask.point(lambda x: 255 if x else 0, '1')

    path = mask_path
    # 保存 mask 图像
    # mask.save(path)
    plt.imsave(path, mask, cmap='Greys_r')
    
    return mask

def points_to_back(points,save_path):
    """
    通过4个关键点真值得到一个不旋转的矩形框。
    Args:
        keypoints: 关键点信息，包含4个点的坐标，形状为(4, 2)
    Returns:
        box: 表示矩形框的四个坐标，形如(x1, y1, x2, y2)
    """
    point = np.copy(points)
    point = point.reshape(-1,2)
    mask = np.zeros((256,256))

    x_max = int(max(point[:, 0]))
    x_min = int(min(point[:, 0]))
    y_max = int(max(point[:, 1]))
    y_min = int(min(point[:, 1]))

    mask[y_min: y_max+1, x_min: x_max+1] = 1
    back = 1-mask
    # print(mask)
    plt.imsave(save_path, back, cmap='Greys_r')
    # n,c,h,w
    return back

def points_to_box(points,save_path):
    """
    通过4个关键点真值得到一个不旋转的矩形框。
    Args:
        keypoints: 关键点信息，包含4个点的坐标，形状为(4, 2)
    Returns:
        box: 表示矩形框的四个坐标，形如(x1, y1, x2, y2)
    """
    point = np.copy(points)
    point = point.reshape(-1,2)
    mask = np.zeros((256,256))

    x_max = int(max(point[:, 0]))
    x_min = int(min(point[:, 0]))
    y_max = int(max(point[:, 1]))
    y_min = int(min(point[:, 1]))

    mask[y_min: y_max+1, x_min: x_max+1] = 1
    plt.imsave(save_path, mask, cmap='Greys_r')
    return mask

def points_to_crop(points,img):
    """
    通过4个关键点真值得到一个不旋转的矩形框。
    Args:
        keypoints: 关键点信息，包含4个点的坐标，形状为(4, 2)
    Returns:
        box: 表示矩形框的四个坐标，形如(x1, y1, x2, y2)
    """
    crop = np.copy(img)
    point = np.copy(points)
    point = point.reshape(-1,2)

    x_max = int(max(point[:, 0]))
    x_min = int(min(point[:, 0]))
    y_max = int(max(point[:, 1]))
    y_min = int(min(point[:, 1]))

    crop_img = crop[y_min:y_max,x_min:x_max,:]
    return crop_img


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
    inter_area = (np.max([0, xx2-xx1+ 1])) * (np.max([0, yy2-yy1+ 1])) #计算交集面积
    iou = inter_area / (area1+area2-inter_area+1e-6) #计算交并比

    return iou

def distance_map(point, pad=10, v=0.15, relax=False):

    x_max = int(max(point[:,0]))
    x_min = int(min(point[:,0]))
    y_max = int(max(point[:,1]))
    y_min = int(min(point[:,1]))

    bbox = [x_min, y_min, x_max, y_max]

    dismap = np.zeros((256, 256))
    dismap = compute_dismap(dismap, bbox)
    return dismap


def compute_dismap(dismap, bbox):
    x_min, y_min, x_max, y_max = bbox[:]

    # draw bounding box
    cv2.line(dismap, (x_min, y_min), (x_max, y_min), color=1, thickness=1)
    cv2.line(dismap, (x_min, y_min), (x_min, y_max), color=1, thickness=1)
    cv2.line(dismap, (x_max, y_max), (x_max, y_min), color=1, thickness=1)
    cv2.line(dismap, (x_max, y_max), (x_min, y_max), color=1, thickness=1)

    tmp = (dismap > 0).astype(np.uint8)  # mark boundary
    tmp_ = deepcopy(tmp)

    fill_mask = np.ones((tmp.shape[0] + 2, tmp.shape[1] + 2)).astype(np.uint8)
    fill_mask[1:-1, 1:-1] = tmp_
    cv2.floodFill(tmp_, fill_mask, (int((x_min + x_max) / 2), int((y_min + y_max) / 2)), 5) # fill pixel inside bounding box

    tmp_ = tmp_.astype(np.int8)
    tmp_[tmp_ == 5] = -1  # pixel inside bounding box
    tmp_[tmp_ == 0] = 1  # pixel on and outside bounding box

    tmp = (tmp == 0).astype(np.uint8)

    dismap = cv2.distanceTransform(tmp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)  # compute distance inside and outside bounding box
    dismap = tmp_ * dismap + 128

    dismap[dismap > 255] = 255
    dismap[dismap < 0] = 0

    dismap = dismap.astype(np.uint8)

    return dismap


def get_dismap(input_image,point):
    _, iH, iW = input_image.shape    
    int_pos = np.uint8(255*np.ones([iH,iW]))

    # for i in range(0,4):
    clk_pos_1 = (int_pos==0)
    clk_pos_2 = (int_pos==0)
    clk_pos_3 = (int_pos==0)
    clk_pos_4 = (int_pos==0)

    clk_pos_1[int(point[0,0]),int(point[0,1])] = 1    
    int_pos_1 = ndimage.distance_transform_edt(1-clk_pos_1)
    int_pos_1 = np.uint8(np.minimum(np.maximum(int_pos_1, 0.0), 255.0))
    cv2.imwrite('xx1.png', int_pos_1)

    clk_pos_2[int(point[1,0]),int(point[1,1])] = 1    
    int_pos_2 = ndimage.distance_transform_edt(1-clk_pos_2)
    int_pos_2 = np.uint8(np.minimum(np.maximum(int_pos_2, 0.0), 255.0))
    cv2.imwrite('xx2.png', int_pos_2)

    clk_pos_3[int(point[2,0]),int(point[2,1])] = 1    
    int_pos_3 = ndimage.distance_transform_edt(1-clk_pos_3)
    int_pos_3 = np.uint8(np.minimum(np.maximum(int_pos_3, 0.0), 255.0))
    cv2.imwrite('xx3.png', int_pos_3)


    clk_pos_4[int(point[3,0]),int(point[3,1])] = 1    
    int_pos_4 = ndimage.distance_transform_edt(1-clk_pos_4)
    int_pos_4 = np.uint8(np.minimum(np.maximum(int_pos_4, 0.0), 255.0))
    cv2.imwrite('xx4.png', int_pos_4)

    int_pos_5 = int_pos_1 / 4 + int_pos_2 / 4 + int_pos_3 / 4 + int_pos_4 / 4
    # int_pos_5 = 255 - int_pos_5
    # cv2.imwrite('xx5.png', int_pos_5)
    # int_pos = torch.
    # int_pos = int_pos_1 + int_pos_2 + int_pos_3 +int_pos_4
    # int_pos = (255*4 - int_pos)/4
    # cv2.imwrite('xx.png', int_pos)

    return torch.tensor(int_pos_5,dtype=torch.float32)

def depthwise_cross_correlation(image, kernel):
    # Get the dimensions of the input image
    image_height, image_width, image_depth = image.shape
    
    # Get the dimensions of the kernel
    kernel_height, kernel_width, kernel_depth = kernel.shape
    
    # Pad the image based on the kernel size
    pad_height = (kernel_height - 1) // 2
    pad_width = (kernel_width - 1) // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    
    # Create an output array to store the results
    output_height = image_height
    output_width = image_width
    output_depth = kernel_depth
    output = np.zeros((output_height, output_width, output_depth))
    
    # Perform the depth-wise cross-correlation
    for i in range(output_height):
        for j in range(output_width):
            for d in range(output_depth):
                output[i, j, d] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width, d] * kernel[:, :, d])
    
    return output
    


def get_bbox(mask, points=None, pad=0, relax=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if relax:
        pad = 0

    x_min_bound = 0
    y_min_bound = 0
    x_max_bound = mask.shape[1] - 1
    y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return [x_min, y_min, x_max, y_max]

def add_turbulence(bbox, v=0.15):
    x_min, y_min, x_max, y_max = bbox[:]
    x_min_new = int(x_min + v * np.random.normal(0, 1) * (x_max - x_min))
    x_max_new = int(x_max + v * np.random.normal(0, 1) * (x_max - x_min))
    y_min_new = int(y_min + v * np.random.normal(0, 1) * (y_max - y_min))
    y_max_new = int(y_max + v * np.random.normal(0, 1) * (y_max - y_min))

    return [x_min_new, y_min_new, x_max_new, y_max_new]


def fixed_resize(sample, resolution, flagval=None):
    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(
            round(float(resolution) / np.min(sample.shape[:2]) * np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)

    return sample


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()


def distance(x1,y1,x2,y2):
    # sigma_i = 0.015
    sigma_i = 0.025
    max_dis = np.sqrt(256**2 + 256 **2)
    dis = np.sqrt(np.abs((x1-x2)**2+(y1-y2)**2)) / max_dis
    if x1 == x2 and y1 == y2:
        dist = 1
    else:
        dist = np.exp((-(dis**2)/(2*sigma_i)))

    return dist

def similar(image,x1,y1,x2,y2):
    sigma_sim = 0.15
    sim = np.abs(image[int(x1)][int(y1)] - image[x2][y2])
    if x1 == x2 and y1 == y2:
        simi = 1
    else:
        simi = np.exp((-(sim) / (2*sigma_sim)))

    return simi

def bilateral_filter(image, kepoint):
    new_image = np.zeros(image.shape)
    # new_image = torch.zeros_like(image)
    for row in range(len(image)):
        for col in range(len(image[0])):

            n_y1 = int(kepoint[0,0])
            n_x1 = int(kepoint[0,1])
            n_y2 = int(kepoint[1,0])
            n_x2 = int(kepoint[1,1])
            n_y3 = int(kepoint[2,0])
            n_x3 = int(kepoint[2,1])
            n_y4 = int(kepoint[3,0])
            n_x4 = int(kepoint[3,1])
            # if n_x >= len(image):
            #     n_x -= len(image)
            # if n_y >= len(image[0]):
            #     n_y -= len(image[0])
            gs1 = distance(n_x1, n_y1, row, col)
            gi1 = similar(image,n_x1, n_y1, row, col)

            gs2 = distance(n_x2, n_y2, row, col)
            gi2 = similar(image,n_x2, n_y2, row, col)

            gs3 = distance(n_x3, n_y3, row, col)
            gi3 = similar(image,n_x3, n_y3, row, col)

            gs4 = distance(n_x4, n_y4, row, col)
            gi4 = similar(image,n_x4, n_y4, row, col)

            wp1 = gi1 * gs1 * 255
            wp2 = gi2 * gs2 * 255
            wp3 = gi3 * gs3 * 255
            wp4 = gi4 * gs4 * 255
            wp = wp1 + wp2 + wp3 + wp4

            new_image[row][col] = wp

    cv2.imwrite("new_image.png", new_image)
    return torch.tensor(new_image,dtype=torch.float32)

def bilateral_filter_multi(image, keypoints, img_path, img_name):
    new_image = np.zeros(image.shape, dtype=np.float32)
    height, width = image.shape

    for y in range(height):
        for x in range(width):
            wp = 0
            for kp in keypoints:
                n_x = int(kp[0])
                n_y = int(kp[1])
                if n_x >= width or n_y >= height:
                    continue

                gs = distance(n_x, n_y, x, y)
                gi = similar(image, n_x, n_y, x, y)

                wp += gi * gs * 255

            new_image[y, x] = wp

    savepath = os.path.join(str(img_path), 'dismap', str(img_name))
    if not os.path.exists(os.path.join(str(img_path), 'dismap')):
        os.makedirs(os.path.join(str(img_path), 'dismap'))
    cv2.imwrite(savepath, new_image)
    return torch.tensor(new_image, dtype=torch.float32)

