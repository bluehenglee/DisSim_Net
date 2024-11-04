import torch
import os
import numpy as np
from torch import nn
import torchvision
import torch.utils.data
from torchvision import datasets, transforms, models
import cv2
import json
from utils import *
from data_pre import one_json_to_numpy,json_to_numpy
from similarity import save_dismap
from concurrent.futures import ThreadPoolExecutor
import time

# box_3D的数据仓库
class Dataset(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img_path = os.path.join(self.dataset_path, 'imgs', self.img_name_list[index])
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        # img = cv2.resize(img, (512, 352))
        img = cv2.resize(img, (256, 256))
        img = transforms.ToTensor()(img)

        # 读入标签
        label_path = os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0]+'.json')
        img_name = self.img_name_list[index]
        masks_path = os.path.join(self.dataset_path, 'masks', img_name)
        # fore_mask = os.path.join(self.dataset_path, 'fore', img_name)
        # back_mask = os.path.join(self.dataset_path, 'back', img_name)
        box_mask = os.path.join(self.dataset_path, 'box', img_name)

        # print(label_path)
        # mask是4个(x,y)分开了的8个数
        mask = one_json_to_numpy(label_path)

        # 计算resize前后图片尺寸的比例

        new_width, new_height = 256, 256  # 调整后的图片尺寸，这里以256*256为例
        width_scale = new_width / width
        height_scale = new_height / height

        # 根据比例调整标签中关键点的坐标
        for i in range(0, len(mask), 2):
            mask[i] = mask[i] * width_scale
            mask[i + 1] = mask[i + 1] * height_scale

        # point_to_fore(mask,fore_mask)
        points_to_box(mask,box_mask)
        # mask = json_to_numpy(label_path)
        # mask = np.load(os.path.join(self.dataset_path, 'masks', self.img_name_list[index].split('.')[0] + '.npy'))
        mask = torch.tensor(mask, dtype=torch.float32)

        # print(img_path)
        # print(label_path)
        # print('-----------------')
        if img_path.split('.')[0] != label_path.split('.')[0]:
            print("数据不一致")

        return img, mask, masks_path
    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)

# boxinst  train3的数据仓库
class Dataset_dis(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img_path = os.path.join(self.dataset_path, 'imgs', self.img_name_list[index])
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        # img = cv2.resize(img, (512, 352))
        img = cv2.resize(img, (256, 256))
        img = transforms.ToTensor()(img)

        image = cv2.imread(img_path,0)
        image = cv2.resize(image, (256, 256))
        image = image / 255
        # image = image.transpose(2, 0, 1)


        # 读入标签
        label_path = os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0]+'.json')
        img_name = self.img_name_list[index]
        masks_path = os.path.join(self.dataset_path, 'masks', img_name)
        # print(label_path)
        # mask是4个(x,y)分开了的8个数
        point = one_json_to_numpy(label_path)

        masks_box = os.path.join(self.dataset_path, 'box', img_name)
        box = cv2.imread(masks_box,0)
        box = transforms.ToTensor()(box)

        # 计算resize前后图片尺寸的比例

        new_width, new_height = 256, 256  # 调整后的图片尺寸，这里以256*256为例
        width_scale = new_width / width
        height_scale = new_height / height

        # 根据比例调整标签中关键点的坐标
        for i in range(0, len(point), 2):
            point[i] = point[i] * width_scale
            point[i + 1] = point[i + 1] * height_scale

        # mask = json_to_numpy(label_path)
        # mask = np.load(os.path.join(self.dataset_path, 'masks', self.img_name_list[index].split('.')[0] + '.npy'))
        point = torch.tensor(point, dtype=torch.float32)

        # print(img_path)
        # print(label_path)
        # print('-----------------')
        if img_path.split('.')[0] != label_path.split('.')[0]:
            print("数据不一致")
        point = point.reshape(-1, 2)   
        # dismap = distance_map(point)
        # dismap = get_dismap(img,point)
        dismap = bilateral_filter(image,point)
        save_dismap(image,point,img_name)
        dismap = dismap / 255

        return img, point, box, masks_path,dismap
    
    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)


class Dataset_f(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img_path = os.path.join(self.dataset_path, 'imgs', self.img_name_list[index])
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        # img = cv2.resize(img, (512, 352))
        img = cv2.resize(img, (256, 256))
        img = transforms.ToTensor()(img)

        # 读入标签
        label_path = os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0]+'.json')
        img_name = self.img_name_list[index]
        masks_path = os.path.join(self.dataset_path, 'masks', img_name)
        fore_path = os.path.join(self.dataset_path, 'fore', img_name)
        gt_path = os.path.join(self.dataset_path, 'gt', img_name)
        box_mask = os.path.join(self.dataset_path, 'box', img_name)

        gt = cv2.imread(masks_path)
        gt = cv2.resize(gt, (256, 256))

        plt.imsave(gt_path, gt, cmap='Greys_r')

        # fore = cv2.imread(fore_path,cv2.IMREAD_GRAYSCALE)
        # fore = transforms.ToTensor()(fore)
        # img = cv2.resize(img, (512, 352))


        fore_mask = os.path.join(self.dataset_path, 'fore', img_name)
        back_mask = os.path.join(self.dataset_path, 'back', img_name)
        # print(label_path)
        # mask是4个(x,y)分开了的8个数
        mask = one_json_to_numpy(label_path)

        # 计算resize前后图片尺寸的比例

        new_width, new_height = 256, 256  # 调整后的图片尺寸，这里以256*256为例
        width_scale = new_width / width
        height_scale = new_height / height

        # 根据比例调整标签中关键点的坐标
        for i in range(0, len(mask), 2):
            mask[i] = mask[i] * width_scale
            mask[i + 1] = mask[i + 1] * height_scale

        point_to_fore(mask,fore_mask)
        points_to_back(mask,back_mask)
        points_to_box(mask,box_mask)
        # mask = json_to_numpy(label_path)
        # mask = np.load(os.path.join(self.dataset_path, 'masks', self.img_name_list[index].split('.')[0] + '.npy'))
        mask = torch.tensor(mask, dtype=torch.float32)

        # print(img_path)
        # print(label_path)
        # print('-----------------')
        if img_path.split('.')[0] != label_path.split('.')[0]:
            print("数据不一致")

        # return img, mask, fore, masks_path
        return img, mask, gt, masks_path
    
    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)
    

    
class Dataset_fb(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img_path = os.path.join(self.dataset_path, 'imgs', self.img_name_list[index])
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        # img = cv2.resize(img, (512, 352))
        img = cv2.resize(img, (256, 256))
        img = transforms.ToTensor()(img)

        # 读入标签
        label_path = os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0]+'.json')
        img_name = self.img_name_list[index]
        masks_path = os.path.join(self.dataset_path, 'masks', img_name)
        fore_path = os.path.join(self.dataset_path, 'fore', img_name)
        back_path = os.path.join(self.dataset_path, 'back', img_name)
        masks_box = os.path.join(self.dataset_path, 'box', img_name)

        fore = cv2.imread(fore_path,cv2.IMREAD_GRAYSCALE)
        fore = transforms.ToTensor()(fore)

        back = cv2.imread(back_path,cv2.IMREAD_GRAYSCALE)
        back = transforms.ToTensor()(back)
        # img = cv2.resize(img, (512, 352))
        box = cv2.imread(masks_box,0)
        box = transforms.ToTensor()(box)

        # fore_mask = os.path.join(self.dataset_path, 'fore', img_name)
        # back_mask = os.path.join(self.dataset_path, 'back', img_name)
        # print(label_path)
        # mask是4个(x,y)分开了的8个数
        mask = one_json_to_numpy(label_path)

        # 计算resize前后图片尺寸的比例

        new_width, new_height = 256, 256  # 调整后的图片尺寸，这里以256*256为例
        width_scale = new_width / width
        height_scale = new_height / height

        # 根据比例调整标签中关键点的坐标
        for i in range(0, len(mask), 2):
            mask[i] = mask[i] * width_scale
            mask[i + 1] = mask[i + 1] * height_scale

        # point_to_fore(mask,fore_mask)
        # points_to_back(mask,back_mask)
        # mask = json_to_numpy(label_path)
        # mask = np.load(os.path.join(self.dataset_path, 'masks', self.img_name_list[index].split('.')[0] + '.npy'))
        mask = torch.tensor(mask, dtype=torch.float32)

        # print(img_path)
        # print(label_path)
        # print('-----------------')
        if img_path.split('.')[0] != label_path.split('.')[0]:
            print("数据不一致")
        
        # 新加的代码
        point = mask.reshape(-1, 2)   
        # dismap = distance_map(point)
        dismap = get_dismap(img,point)
        dismap = dismap / 255

        return img, box, fore, back, masks_path, dismap
    # 数据集的大小

    def __len__(self):
        return len(self.img_name_list)
    
# box_3D的数据仓库
class Dataset_cluster(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img_path = os.path.join(self.dataset_path, 'imgs', self.img_name_list[index])
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        # img = cv2.resize(img, (512, 352))
        img = cv2.resize(img, (256, 256))
        img = transforms.ToTensor()(img)

        image = cv2.imread(img_path,0)
        image = cv2.resize(image, (256, 256))
        image = image / 255
        # image = image.transpose(2, 0, 1)

        # 读入标签
        label_path = os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0]+'.json')
        img_name = self.img_name_list[index]
        masks_path = os.path.join(self.dataset_path, 'masks', img_name)

        # mask是4个(x,y)分开了的8个数
        point = one_json_to_numpy(label_path)

        fore_path = os.path.join(self.dataset_path, 'fore', img_name)
        back_path = os.path.join(self.dataset_path, 'back', img_name)
        masks_box = os.path.join(self.dataset_path, 'box', img_name)

        fore = cv2.imread(fore_path,cv2.IMREAD_GRAYSCALE)
        fore = transforms.ToTensor()(fore)

        back = cv2.imread(back_path,cv2.IMREAD_GRAYSCALE)
        back = transforms.ToTensor()(back)

        box = cv2.imread(masks_box,0)
        box = transforms.ToTensor()(box)

        # 计算resize前后图片尺寸的比例

        new_width, new_height = 256, 256  # 调整后的图片尺寸，这里以256*256为例
        width_scale = new_width / width
        height_scale = new_height / height

        # 根据比例调整标签中关键点的坐标
        for i in range(0, len(point), 2):
            point[i] = point[i] * width_scale
            point[i + 1] = point[i + 1] * height_scale

        # mask = json_to_numpy(label_path)
        # mask = np.load(os.path.join(self.dataset_path, 'masks', self.img_name_list[index].split('.')[0] + '.npy'))
        point = torch.tensor(point, dtype=torch.float32)

        # print(img_path)
        # print(label_path)
        # print('-----------------')
        if img_path.split('.')[0] != label_path.split('.')[0]:
            print("数据不一致")
        point = point.reshape(-1, 2)   
        # dismap = distance_map(point)
        # dismap = get_dismap(img,point)
        dismap = bilateral_filter(image,point)
        dismap = dismap / 255

        return img, box, fore, back, masks_path, dismap
    
    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)

# boxinst  train3的数据仓库
class Dataset_dissim(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img_path = os.path.join(self.dataset_path, 'imgs', self.img_name_list[index])
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        # img = cv2.resize(img, (512, 352))
        img = cv2.resize(img, (256, 256))


        image = cv2.imread(img_path,0)
        image = cv2.resize(image, (256, 256))
        image = image / 255
        # image = image.transpose(2, 0, 1)


        # 读入标签
        label_path = os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0]+'.json')
        img_name = self.img_name_list[index]
        masks_path = os.path.join(self.dataset_path, 'masks', img_name)
        # print(label_path)
        # mask是4个(x,y)分开了的8个数
        point = one_json_to_numpy(label_path)

        masks_box = os.path.join(self.dataset_path, 'box', img_name)
        box = cv2.imread(masks_box,0)
        box = transforms.ToTensor()(box)
        # 计算resize前后图片尺寸的比例
        new_width, new_height = 256, 256  # 调整后的图片尺寸，这里以256*256为例
        width_scale = new_width / width
        height_scale = new_height / height

        # 根据比例调整标签中关键点的坐标
        for i in range(0, len(point), 2):
            point[i] = point[i] * width_scale
            point[i + 1] = point[i + 1] * height_scale

        # mask = json_to_numpy(label_path)
        # mask = np.load(os.path.join(self.dataset_path, 'masks', self.img_name_list[index].split('.')[0] + '.npy'))
        point = torch.tensor(point, dtype=torch.float32)
        #crop出前景区域
        crop = points_to_crop(point,img)
        crop = cv2.resize(crop, (256, 256))

        img = transforms.ToTensor()(img)
        crop = transforms.ToTensor()(crop)
        # print(img_path)
        # print(label_path)
        # print('-----------------')
        if img_path.split('.')[0] != label_path.split('.')[0]:
            print("数据不一致")
        point = point.reshape(-1, 2)   
        # dismap = distance_map(point)
        # dismap = get_dismap(img,point)
        dismap = bilateral_filter(image,point)
        # save_dismap(image,point,img_name)
        dismap = dismap / 255

        return img, point, box, masks_path,dismap,crop
    
    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)
    
# box_3D的数据仓库
class Dataset_all(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
        self.to_tensor = transforms.ToTensor()
        # Precompute paths to avoid repeated os.path.join calls
        self.img_paths = [os.path.join(dataset_path, 'imgs', img_name) for img_name in self.img_name_list]
        self.label_paths = [os.path.join(dataset_path, 'labels', img_name.split('.')[0]+'.json') for img_name in self.img_name_list]
        self.fore_paths = [os.path.join(dataset_path, 'fore', img_name) for img_name in self.img_name_list]
        self.back_paths = [os.path.join(dataset_path, 'back', img_name) for img_name in self.img_name_list]
        self.box_paths = [os.path.join(dataset_path, 'box', img_name) for img_name in self.img_name_list]
        self.dismap_paths = [os.path.join(dataset_path, 'dismap', img_name) for img_name in self.img_name_list]
        self.point_paths = [os.path.join(dataset_path, 'point', img_name) for img_name in self.img_name_list]
        self.ellipse_paths = [os.path.join(dataset_path, 'ellipse', img_name) for img_name in self.img_name_list]
        self.scribbles_paths = [os.path.join(dataset_path, 'scribbles', img_name) for img_name in self.img_name_list]
        self.gt_paths = [os.path.join(dataset_path, 'gt', img_name) for img_name in self.img_name_list]

    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # Process image
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        # 调整形状为 (3, 256, 256)
        img = cv2.resize(img, (256, 256))/255.0
        img = img.transpose((2, 0, 1))
        # img = self.to_tensor(img)

        box_path = self.box_paths[index]
        box = cv2.imread(box_path,0)/255.0
        box = np.expand_dims(box, axis=0)
        # box = self.to_tensor(box)

        fore_path = self.fore_paths[index]
        fore = cv2.imread(fore_path,0)/255.0
        fore = np.expand_dims(fore, axis=0)

        # fore = self.to_tensor(fore)

        back_path = self.back_paths[index]
        back = cv2.imread(back_path,0)/255.0
        back = np.expand_dims(back, axis=0)

        # back = self.to_tensor(back)

        gt_path = self.gt_paths[index]
        gt = cv2.imread(gt_path,0)/255.0
        gt = np.expand_dims(gt, axis=0)

        # gt = self.to_tensor(gt)

        dismap_path = self.dismap_paths[index]
        dismap = cv2.imread(dismap_path,0)/255.0
        dismap = np.expand_dims(dismap, axis=0)

        # point_path = self.point_paths[index]
        # point = cv2.imread(point_path,0)/255.0
        # point = np.expand_dims(point, axis=0)

        # dismap = self.to_tensor(dismap)

        # ellipse_path = self.ellipse_paths[index]
        # ellipse = cv2.imread(ellipse_path,0)/255.0
        # ellipse = np.expand_dims(ellipse, axis=0)

        # ellipse = self.to_tensor(ellipse)

        # scribbles_path = self.scribbles_paths[index]
        # scribbles = cv2.imread(scribbles_path,0)/255.0
        # scribbles = np.expand_dims(scribbles, axis=0)

        # scribbles = self.to_tensor(scribbles)

        # return img, box, fore, back, gt, self.gt_paths[index], dismap, ellipse, scribbles
        return img, box, fore, back, gt, self.gt_paths[index], dismap
        # return img, box, fore, back, gt, self.gt_paths[index], point

    def __len__(self):
        return len(self.img_name_list)
 
 # box_3D的数据仓库
class Dataset_sam(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
        self.to_tensor = transforms.ToTensor()
        # Precompute paths to avoid repeated os.path.join calls
        self.img_paths = [os.path.join(dataset_path, 'imgs', img_name) for img_name in self.img_name_list]
        self.label_paths = [os.path.join(dataset_path, 'labels', img_name.split('.')[0]+'.json') for img_name in self.img_name_list]
        self.fore_paths = [os.path.join(dataset_path, 'fore', img_name) for img_name in self.img_name_list]
        self.back_paths = [os.path.join(dataset_path, 'back', img_name) for img_name in self.img_name_list]
        self.box_paths = [os.path.join(dataset_path, 'box', img_name) for img_name in self.img_name_list]
        self.dismap_paths = [os.path.join(dataset_path, 'dismap', img_name) for img_name in self.img_name_list]
        self.point_paths = [os.path.join(dataset_path, 'point', img_name) for img_name in self.img_name_list]
        self.ellipse_paths = [os.path.join(dataset_path, 'ellipse', img_name) for img_name in self.img_name_list]
        self.scribbles_paths = [os.path.join(dataset_path, 'scribbles', img_name) for img_name in self.img_name_list]
        self.gt_paths = [os.path.join(dataset_path, 'gt', img_name) for img_name in self.img_name_list]
        self.sam_paths = [os.path.join(dataset_path, 'SAM_result', img_name) for img_name in self.img_name_list]


    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # Process image
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        # 调整形状为 (3, 256, 256)
        img = cv2.resize(img, (256, 256))/255.0
        img = img.transpose((2, 0, 1))
        # img = self.to_tensor(img)

        box_path = self.box_paths[index]
        box = cv2.imread(box_path,0)/255.0
        box = np.expand_dims(box, axis=0)
        # box = self.to_tensor(box)

        fore_path = self.fore_paths[index]
        fore = cv2.imread(fore_path,0)/255.0
        fore = np.expand_dims(fore, axis=0)

        back_path = self.back_paths[index]
        back = cv2.imread(back_path,0)/255.0
        back = np.expand_dims(back, axis=0)

        gt_path = self.gt_paths[index]
        gt = cv2.imread(gt_path,0)/255.0
        gt = cv2.resize(gt, (256, 256))
        gt = np.expand_dims(gt, axis=0)

        sam_path = self.sam_paths[index]
        sam = cv2.imread(sam_path,0)
        sam = cv2.resize(sam, (256, 256))
        sam = np.expand_dims(sam, axis=0)

        # Create binary masks
        # fore_sam = np.where((fore >= 0.5) & (sam >= 0.5), 1, 0)
        # back_sam = np.where((back >= 0.5) & (sam <= 0.5), 1, 0)
        fore_sam = np.where((sam >= 0.5), 1, 0)
        back_sam = np.where((sam < 0.5), 1, 0)

        dismap_path = self.dismap_paths[index]
        dismap = cv2.imread(dismap_path,0)/255.0
        dismap = np.expand_dims(dismap, axis=0)

        # return img, box, fore, back, gt, self.gt_paths[index], dismap, ellipse, scribbles
        return img, box, fore_sam, back_sam, gt, self.gt_paths[index], dismap
        # return img, box, fore, back, gt, self.gt_paths[index], point

    def __len__(self):
        return len(self.img_name_list)

class Dataset_multi(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img_path = os.path.join(self.dataset_path, 'imgs', self.img_name_list[index])
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        # img = cv2.resize(img, (512, 352))
        img = cv2.resize(img, (256, 256))


        image = cv2.imread(img_path,0)
        image = cv2.resize(image, (256, 256))
        image = image / 255
        # image = image.transpose(2, 0, 1)


        # 读入标签
        label_path = os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0]+'.json')
        # print(label_path)
        # mask是4个(x,y)分开了的8个数
        point_groups = json_to_numpy(label_path)

        img_name = self.img_name_list[index]
        masks_path = os.path.join(self.dataset_path, 'gt', img_name)
        gt = cv2.imread(masks_path,0)
        gt = transforms.ToTensor()(gt)


        masks_box = os.path.join(self.dataset_path, 'box', img_name)
        box = cv2.imread(masks_box,0)
        box = transforms.ToTensor()(box)


        fore_path = os.path.join(self.dataset_path, 'fore', img_name)
        fore = cv2.imread(fore_path,cv2.IMREAD_GRAYSCALE)
        fore = transforms.ToTensor()(fore)


        # 计算resize前后图片尺寸的比例
        new_width, new_height = 256, 256  # 调整后的图片尺寸，这里以256*256为例
        width_scale = new_width / width
        height_scale = new_height / height


        for points in point_groups:
            for i in range(0, len(points), 2):
                points[i] = points[i] * width_scale
                points[i + 1] = points[i + 1] * height_scale
                # point = torch.tensor(point, dtype=torch.float32)

        # 初始化一个空数组来存储所有关键点
        all_points = []
        for points in point_groups:
            points = points.reshape(-1, 2)
            all_points.extend(points)
        # 转换为NumPy数组
        all_points = np.array(all_points)

        # 计算dismap
        dismap = bilateral_filter_multi(image, all_points, self.dataset_path, img_name)
        # 正规化dismap，np
        # dismap = dismap / 255
        # dismap = np.clip(dismap, 0, 1)
        # 正规化dismap，torch
        dismap = dismap / 255
        dismap = torch.clamp(dismap, 0, 1)
        img = transforms.ToTensor()(img)

        return img, gt, fore, box, masks_path, dismap
    
    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)