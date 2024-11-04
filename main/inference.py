import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import cv2
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter
from utils.metrics import *
import copy
from models.PAplusNet import PAplusNet_unet, PAplusNet_densenet, PAplusNet_cenet
from torch.utils.data import Dataset, DataLoader, random_split
import random
import logging
from utils.plot import *
import argparse
from utils.dataset import *
import math
import torch.nn.functional as F

os.chdir(sys.path[0])

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--backbone", type=str, help="unet/densenet/cenet", default="auto")
    parse.add_argument("--epochs", type=int,  default=100)
    parse.add_argument("--train", type=str, help='tn3k/DDTI', default='DDTI')
    parse.add_argument("--batch_size", type=int, default=16)
    parse.add_argument("--lama", type=float, default=1.0)
    parse.add_argument("--gama", type=float, default=0.05)
    parse.add_argument("--patchsize", type=str, default='3')
    parse.add_argument('--DDTIfile', default='/data16t/usr/PAplusNet/data/DDTI/DDTI.json')
    parse.add_argument("--predict_output_path", type=str, default='/data16t/usr/PAplusNet/inference')
    parse.add_argument("--weight_save_path", type=str, default='/data16t/usr/PAplusNet/weights_new')
    parse.add_argument("--writer_dir", type=str, default='/data16t/usr/PAplusNet/runs/inference')
    parse.add_argument('--dataset', default='DDTI',  # dsb2018_256
                       help='dataset name:DDTI/tn3k/TRFE/ZY')
    parse.add_argument("--log_dir", default='../log', help="log dir")
    parse.add_argument("--threshold", type=float, default=0.5)

    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch, str(args.dataset))
    filename = dirname +'log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def split_dataset(dataset, val_split=0.2, test_split=0.1):
    dataset_len = len(dataset)
    val_len = int(dataset_len * val_split)
    test_len = int(dataset_len * test_split)
    train_len = dataset_len - val_len - test_len
    return random_split(dataset, [train_len, val_len, test_len])

def load_split_indices(filename):
    with open(filename, 'r') as f:
        indices = json.load(f)
    return indices

def create_dataloaders(dataset, indices, batch_size, num_workers=4):
    train_indices = indices['train']
    val_indices = indices['val']

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader

def get_models(args):
    if args.backbone == 'unet':
        model = PAplusNet_unet(4,1).to(device)
    elif args.backbone == 'densenet':
        model = PAplusNet_densenet(in_channel=4, num_classes=1).to(device)
    elif args.backbone == 'cenet':
        model = PAplusNet_cenet(4,1).to(device)
    return model

# load model weights
def load_model_weights(model, weight_path):
    checkpoint = torch.load(weight_path)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

def calculate_metrics(predict, gt):
    """
    Calculate various metrics including mIoU, Dice, Precision, and Hausdorff Distance.

    Args:
    predict (torch.Tensor): Prediction tensor, binary (0 or 1) on GPU.
    gt (torch.Tensor): Ground truth tensor, binary (0 or 1) on GPU.

    Returns:
    dict: A dictionary containing mIoU, Dice, Precision, and Hausdorff Distance.
    """
    # Calculate intersection and union for IoU
    intersection = torch.logical_and(predict, gt).sum().float()
    union = torch.logical_or(predict, gt).sum().float()
    miou = (intersection / union) if union != 0 else torch.tensor(0.0, device=predict.device)

    # Calculate Dice coefficient
    dice = (2. * intersection / (predict.sum() + gt.sum())) if (predict.sum() + gt.sum()) != 0 else torch.tensor(0.0, device=predict.device)

    # Calculate Precision
    precision = (intersection / predict.sum()) if predict.sum() != 0 else torch.tensor(0.0, device=predict.device)

    # Gather all metrics in a dictionary
    metrics = {
        'mIoU': miou.item(),  # Convert to Python float for easier handling outside Torch
        'Dice': dice.item(),
        'Precision': precision.item()
        # 'Hausdorff Distance': hd
    }

    return metrics

def inference(args, model, val_dataloaders):
    model = model.eval()
    iou_list = []
    dice_list = []
    hd_list = []
    precision_list = []
    with torch.no_grad():
        num = len(val_dataloaders)
        for index, (x, box, fore, back, gt, masks_path, dismap) in enumerate(val_data_loader):
            img, box, fore, back, gt, dismap = map(
                lambda t: t.to(device, non_blocking=True, dtype=torch.float32), 
                (x, box, fore, back, gt, dismap)
            )
            img_input = torch.cat([img, dismap], dim=1)

            seg, loss_contrast = model(img_input, fore, back)

            masks = masks_path    #picture path
            img_y = seg.cpu().detach().numpy()  #[4,256,256]
            img_y = img_y.squeeze()
            img_y[img_y < 0.6] = 0

            predict_output_path = os.path.join(str(args.predict_output_path), 'PAplusNet' + '_' + str(args.backbone), str(args.train), str(args.dataset)+'_'+str(args.lama)+'_'+str(args.gama))
            if not os.path.exists(predict_output_path):
                os.makedirs(predict_output_path)
            plt.imsave(predict_output_path + '/' + masks[0].split('/')[-1], img_y, cmap='Greys_r')

            predictions_binary = (seg >= args.threshold).float()
            gt = (gt >= args.threshold).float()
            metrics = calculate_metrics(predictions_binary, gt)
            tem_hd = metrics['Hausdorff Distance'] #mask[0] is a path
            tem_iou = metrics['mIoU']
            tem_dice =metrics['Dice']
            tem_precision = metrics['Precision']
            print('tem_hd=%f,temp_iou=%f,tem_dice=%f,tem_precision=%f' % (tem_hd,tem_iou,tem_dice,tem_precision))
            
            iou_list.append(tem_iou)
            dice_list.append(tem_dice)
            hd_list.append(tem_hd)
            precision_list.append(tem_precision)

            miou_array = np.array(iou_list)
            aver_iou = np.mean(miou_array)
            std_miou = np.std(miou_array)

            dice_array = np.array(dice_list)
            aver_dice = np.mean(dice_array)
            std_dice = np.std(dice_array)

            hd_array = np.array(hd_list)
            aver_hd = np.mean(hd_array)
            std_hd= np.std(hd_array)

            precision_array = np.array(precision_list)
            aver_precision = np.mean(precision_array)
            std_precision = np.std(precision_array)

        print('##################')
        print('# ---- Mean ---- #')
        print('##################')

        print(str(args.arch)+'_'+str(args.train)+'_'+str(args.dataset)+'_'+str(args.mode))
        print('Miou=%f,aver_hd=%f,aver_dice=%f,aver_precision=%f' % (aver_iou,aver_hd,aver_dice,aver_precision))
        print('std_iou=%f,std_hd=%f,std_dice=%f,std_precision=%f' % (std_miou,std_hd,std_dice,std_precision))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f,aver_precision=%f' % (aver_iou,aver_hd,aver_dice,aver_precision))
    

    
if __name__ == "__main__":
    f = torch.cuda.is_available()
    device = torch.device("cuda" if f else "cpu")
    args = getArgs()
    model = get_models(args)
    weight_path = os.path.join(args.weight_save_path, str(args.arch) + '_' + str(args.dataset)+'_'+str(args.lama)+'_'+str(args.gama)+'.pth')
    print(weight_path)
    load_model_weights(model, weight_path)

    if args.dataset == 'tn3k':
        val_dataset = Dataset_all(os.path.join('..', 'data', str(args.dataset), 'test'))#TRFE
        val_data_loader = DataLoader(dataset=val_dataset,
                                        batch_size= 1,
                                        shuffle=False,
                                        num_workers=4)
    elif args.dataset == 'DDTI':
        dataset = Dataset_all(os.path.join('..', 'data', str(args.dataset)))
        # 加载索引
        indices = load_split_indices(args.DDTIfile)
        # 创建 DataLoader
        train_data_loader, val_data_loader = create_dataloaders(dataset, indices, args.batch_size)

    elif args.dataset == 'ZY' or (args.dataset == 'DDTI' and args.train == 'TRFE'):
        val_dataset = Dataset_all(os.path.join('..', 'data', str(args.dataset)))#ZY、DDTI
        val_data_loader = DataLoader(dataset=val_dataset,
                                        batch_size= 1,
                                        shuffle=False,
                                        num_workers=4)
    inference(args,model,val_data_loader)


