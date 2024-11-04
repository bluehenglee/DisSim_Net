import torch
import os
import numpy as np
from torch import nn
import torchvision
import json
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import cv2
from utils_data import *
import argparse
from practical_function import unfold_wo_center,get_images_color_similarity,compute_pairwise_term

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_name", type=str, help="dataset for preprocessing,DDTI/ZY/tn3ksl/TRFE/", default='DDTI')
    args = parse.parse_args()
    return args

def one_json_to_numpy(dataset_path):
    with open(dataset_path) as fp:
        json_data = json.load(fp)
        points = json_data['shapes']

    # print(points)
    landmarks = []
    for point in points:
        # for p in point['points'][0]:
        # p是一个数一个数存的，xy坐标不在一起，landmarks是一维列表
        for p in point['points'][0]:
            landmarks.append(p)

    # print(landmarks)
    landmarks = np.array(landmarks)
    # 把8个数变成4对坐标,可以不加
    # landmarks = landmarks.reshape(-1, 2)

    # 保存为np
    # np.save(os.path.join(save_path, name.split('.')[0] + '.npy'), landmarks)
    return landmarks

def json_to_numpy(dataset_path):
    with open(dataset_path, 'r') as fp:
        json_data = json.load(fp)
        shapes = json_data['shapes']

    # 初始化一个列表来存储所有关键点组
    keypoint_groups = []

    # 遍历shapes列表，每四个点构成一个组
    for i in range(0, len(shapes), 4):
        # 检查是否有足够的点来构成一个关键点组
        if i + 3 < len(shapes):
            # 提取四个点
            points = [shape['points'] for shape in shapes[i:i+4]]
            # 将每个点的坐标从列表转换为浮点数
            points = [[float(coord) for coord in point] for sublist in points for point in sublist]
            # 将点的列表转换为NumPy数组，并添加到关键点组列表中
            keypoint_group = np.array(points)
            keypoint_groups.append(keypoint_group)

    # 将所有关键点组合并为一个NumPy数组
    keypoint_groups_array = np.array(keypoint_groups)

    return keypoint_groups_array

def create_folder(args):
    if args.data_name == 'tn3k':
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'train','box')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'train','box'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'train','gt')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'train','gt'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'train','fore')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'train','fore'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'train','back')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'train','back'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'train','dismap')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'train','dismap'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','box')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','box'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','gt')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','gt'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','fore')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','fore'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','back')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','back'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','dismap')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','dismap'))
    elif args.data_name == 'DDTI':
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'box')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'box'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'gt')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'gt'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'fore')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'fore'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'back')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'back'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'dismap')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'dismap'))

class Dataset_processing(torch.utils.data.Dataset):
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
        width, height = img.shape[1], img.shape[0]#h,w,c高度（rows）y,宽度（columns)x,通道数（channels)c
        # img = cv2.resize(img, (512, 352))
        img = cv2.resize(img, (256, 256))
        # img = transforms.ToTensor()(img)

        image = cv2.imread(img_path,0)#h,w 高度（rows）y,宽度（columns)x
        image = cv2.resize(image, (256, 256))
        image = image / 255

        # 读入标签
        label_path = os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0]+'.json')
        img_name = self.img_name_list[index]
        # only use in validation
        masks_path = os.path.join(self.dataset_path, 'masks', img_name)
        gt_path = os.path.join(self.dataset_path, 'gt', img_name)
        gt = cv2.imread(masks_path,0)
        gt = cv2.resize(gt, (256, 256))
        plt.imsave(gt_path, gt, cmap='Greys_r')
        box_mask = os.path.join(self.dataset_path, 'box', img_name)
        fore_mask = os.path.join(self.dataset_path, 'fore', img_name)
        back_mask = os.path.join(self.dataset_path, 'back', img_name)
        dismap = os.path.join(self.dataset_path, 'dismap', img_name)
        point_image = os.path.join(self.dataset_path, 'point', img_name)

        point_groups = json_to_numpy(label_path)
        # 计算resize前后图片尺寸的比例
        new_width, new_height = 256, 256  # 调整后的图片尺寸，这里以256*256为例
        width_scale = new_width / width
        height_scale = new_height / height

        # 根据比例调整标签中关键点的坐标
        point_groups[:, :, 0] *= width_scale  # Scale x coordinates
        point_groups[:, :, 1] *= height_scale # Scale y coordinates

        fore = point_to_fore(point_groups,fore_mask)
        back = points_to_back(point_groups,back_mask)
        box = points_to_box(point_groups,box_mask)
        

        point_groups = torch.tensor(point_groups, dtype=torch.float32)
        print(point_groups.shape)
        # 初始化一个空数组来存储所有关键点
        all_points = []
        for points in point_groups:
            points = points.reshape(-1, 2)
            all_points.extend(points)
        # 转换为NumPy数组
        all_points = np.array(all_points)

        print(all_points.shape)
        # 计算dismap
        dismap = bilateral_filter_multi(image, all_points, self.dataset_path, img_name)
        print("save dismap")
        
        if img_path.split('.')[0] != label_path.split('.')[0]:
            print("数据不一致")

        return img, masks_path
        # return img, box, fore, back, masks_path
    
    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)

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

        return img, box, fore, back, masks_path

    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)

if __name__ =="__main__":
    f = torch.cuda.is_available()
    device = torch.device("cuda" if f else "cpu")
    args = getArgs()
    create_folder(args)
    #train and test


    if args.data_name == 'tn3k':
        train_dataset = Dataset_processing(os.path.join('.', 'data', str(args.data_name), 'train'))#tn3k
        val_dataset = Dataset_processing(os.path.join('.', 'data', str(args.data_name), 'test'))#tn3k
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=8,
                                                        shuffle=True)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                        batch_size=8,
                                                        shuffle=True)
        for index,(img, masks_path) in enumerate(train_data_loader):
            print('preprocessing path data:'%(masks_path))
            
        for index,(img, masks_path) in enumerate(val_data_loader):
            print('preprocessing valid data:'%(masks_path))


    elif args.data_name == 'DDTI':
        dataset = Dataset_processing(os.path.join('.', 'data', str(args.data_name)))
        all_data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=8,
                                                        shuffle=True)
        for index,(img, masks_path) in enumerate(all_data_loader):
            print('preprocessing:'%(masks_path))
