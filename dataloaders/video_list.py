import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from glob import glob
import os.path as osp
import pdb
from mypath import Path
import cv2
import copy

# several data augumentation strategies
def cv_random_flip(imgs, imgs_ycbcr, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(len(imgs_ycbcr)):
            imgs_ycbcr[i] = imgs_ycbcr[i].transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, imgs_ycbcr, label

def randomCrop(imgs, imgs_ycbcr, label):
    border = 30
    image_width = imgs[0].size[0]
    image_height = imgs[0].size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    
    for i in range(len(imgs)):
        imgs[i] = imgs[i].crop(random_region)
    for i in range(len(imgs_ycbcr)):
        imgs_ycbcr[i] = imgs_ycbcr[i].crop(random_region)
        
    return imgs, imgs_ycbcr, label.crop(random_region)

def randomRotation(imgs, imgs_ycbcr, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        for i in range(len(imgs)):
            imgs[i] = imgs[i].rotate(random_angle, mode)
        for i in range(len(imgs_ycbcr)):
            imgs_ycbcr[i] = imgs_ycbcr[i].rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return imgs, imgs_ycbcr, label

def colorEnhance(imgs, imgs_ycbcr):
    for i in range(len(imgs)-3):
        bright_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        imgs[i] = ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        imgs[i] = ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
    for i in range(len(imgs_ycbcr)):
        bright_intensity = random.randint(5, 15) / 10.0
        imgs[i+len(imgs)-3] = ImageEnhance.Brightness(imgs[i+len(imgs)-3]).enhance(bright_intensity)
        imgs_ycbcr[i] = ImageEnhance.Brightness(imgs_ycbcr[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        imgs[i+len(imgs)-3] = ImageEnhance.Contrast(imgs[i+len(imgs)-3]).enhance(contrast_intensity)
        imgs_ycbcr[i] = ImageEnhance.Contrast(imgs_ycbcr[i]).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        imgs[i+len(imgs)-3] = ImageEnhance.Color(imgs[i+len(imgs)-3]).enhance(color_intensity)
        imgs_ycbcr[i] = ImageEnhance.Color(imgs_ycbcr[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        imgs[i+len(imgs)-3] = ImageEnhance.Sharpness(imgs[i+len(imgs)-3]).enhance(sharp_intensity)
        imgs_ycbcr[i] = ImageEnhance.Sharpness(imgs_ycbcr[i]).enhance(sharp_intensity)
    return imgs, imgs_ycbcr

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255       
    return Image.fromarray(img)

def generate_point(img):
    original_size = img.shape
    if img.max() == 0:
        return np.array([0, 0, 0, 0]), original_size
    contours, _ = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contour = max(contours, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(contour)
    box = [x, y, x+w, y+h]

    return np.array(box), original_size

class VideoDataset(data.Dataset):
    def __init__(self, dataset='MoCA', trainsize=256, window_length=10, split='MoCA-Video-Train'):
        self.trainsize = trainsize
        self.window_length = window_length
        self.image_list = []
        self.gt_list = []
        self.extra_info = []

        if dataset == 'MoCA': 
            root = Path.db_root_dir('MoCA')
            img_format = '*.jpg'
            
        elif dataset == 'CAD2016':    
            root = Path.db_root_dir('CAD2016')
            img_format = '*.png'

        data_root = osp.join(root, split)

        for scene in os.listdir(osp.join(data_root)):
            if split=='MoCA-Video-Train':
                images  = sorted(glob(osp.join(data_root, scene, 'Frame', img_format)))
            elif split=='TrainDataset_per_sq':
                images  = sorted(glob(osp.join(data_root, scene, 'Imgs', img_format)))
            gt_list = sorted(glob(osp.join(data_root, scene, 'GT', '*.png')))
            # pdb.set_trace()

            begin_frame = self.window_length - 1
            for i in range(begin_frame, len(images)):
                self.extra_info += [(scene, i)]
                self.gt_list += [gt_list[i]]
                self.image_list += [images[i - begin_frame : i+1]]

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        imgs = []
        imgs_ycbcr = []
        names= []
        index = index % len(self.image_list)

        for i in range(len(self.image_list[index])):
            imgs += [self.rgb_loader(self.image_list[index][i])]
        for i in range(len(self.image_list[index])-3, len(self.image_list[index])):  
            imgs_ycbcr += [self.ycbcr_loader(self.image_list[index][i])]

        scene = self.image_list[index][-1].split('/')[-3] 
        name = self.image_list[index][-1].split('/')[-1]
        gt = self.binary_loader(self.gt_list[index])
        
        imgs, imgs_ycbcr, gt = cv_random_flip(imgs, imgs_ycbcr, gt)
        imgs, imgs_ycbcr, gt = randomCrop(imgs, imgs_ycbcr, gt)
        imgs, imgs_ycbcr, gt = randomRotation(imgs, imgs_ycbcr, gt)
        imgs, imgs_ycbcr = colorEnhance(imgs, imgs_ycbcr)
        gt = randomPeper(gt)
        imgs_sam = copy.deepcopy(imgs[-1])
        for i in range(len(imgs)):
            imgs[i] = self.img_transform(imgs[i])
        for i in range(len(imgs_ycbcr)):
            imgs_ycbcr[i] = self.img_transform(imgs_ycbcr[i])
        imgs_sam = self.transform(imgs_sam) * 255.0
        gt = self.transform(gt)
        
        point, _ = generate_point(np.array(gt[0, :, :] * 255).astype(np.uint8))
        point = point / (self.trainsize, self.trainsize, self.trainsize, self.trainsize)
        
        return imgs, imgs_ycbcr, imgs_sam, gt, point

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L') 
            
    def ycbcr_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('YCbCr')

    def __len__(self):
        return len(self.image_list)

# dataloader for training
def get_loader(dataset, batchsize, trainsize, train_split, window_length,
    shuffle=True, num_workers=12, pin_memory=False):
    dataset = VideoDataset(dataset, trainsize, window_length, split=train_split)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, dataset='MoCA', split='TestDataset_per_sq', testsize=256):
        self.testsize = testsize
        self.image_list = []
        self.gt_list = []

        if dataset == 'CAD2016':    
            root = Path.db_root_dir('CAD2016')
            img_format = '*.png'

            for scene in os.listdir(osp.join(root)):
                images  = sorted(glob(osp.join(root, scene, 'frames', img_format)))
                gt_list = sorted(glob(osp.join(root, scene, 'pseudo', '*.png')))
                
                begin_frame = 2
                for i in range(begin_frame, len(images)):
                    self.gt_list    += [ gt_list[i] ]
                    self.image_list += [ [images[i-2], 
                                       images[i-1], 
                                       images[i]] ]

        else: 
            root = Path.db_root_dir('MoCA')
            img_format = '*.jpg'
            data_root = osp.join(root, split)

            for scene in sorted(os.listdir(osp.join(data_root))):
                if split=='MoCA-Video-Test' or split=='MoCA-Video-Train':
                    images  = sorted(glob(osp.join(data_root, scene, 'Frame', img_format)))
                elif split=='TestDataset_per_sq':
                    images  = sorted(glob(osp.join(data_root, scene, 'Imgs', img_format)))
                gt_list = sorted(glob(osp.join(data_root, scene, 'GT', '*.png')))

                begin_frame = 2
                for i in range(begin_frame, len(images)):
                    self.gt_list    += [ gt_list[i] ]
                    self.image_list += [ [images[i-2], 
                                       images[i-1], 
                                       images[i]] ]

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.img_sam_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])

        self.index = 0
        self.size = len(self.gt_list)

    def load_data(self):
        imgs = []
        imgs_sam = []
        imgs_ycbcr = []

        for i in range(len(self.image_list[self.index])):
            imgs += [self.rgb_loader(self.image_list[self.index][i])]
            imgs_sam += [self.rgb_loader(self.image_list[self.index][i])]
            imgs_ycbcr += [self.ycbcr_loader(self.image_list[self.index][i])]
            
            imgs[i] = self.img_transform(imgs[i]).unsqueeze(0)
            imgs_sam[i] = (self.img_sam_transform(imgs_sam[i]) * 255.0).unsqueeze(0)
            imgs_ycbcr[i] = self.img_transform(imgs_ycbcr[i]).unsqueeze(0)
            
        scene = self.image_list[self.index][-1].split('/')[-3]  
        name = self.image_list[self.index][-1].split('/')[-1]
        gt = self.binary_loader(self.gt_list[self.index])
        gt = self.transform(gt)
        
        points, original_size = generate_point(np.array(gt[0, :, :] * 255).astype(np.uint8))
        points = points / (original_size[1], original_size[0], original_size[1], original_size[0])
        
        self.index += 1
        self.index = self.index % self.size
    
        return imgs, imgs_ycbcr, imgs_sam, points, gt, name, scene

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
            
    def ycbcr_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('YCbCr')
            
    def __len__(self):
        return self.size
    
class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        
        self.image_path, self.label_path = [], []
        
        lst_pred = sorted(os.listdir(img_root))
        for l in lst_pred:
            self.image_path.append(osp.join(img_root, l))
            self.label_path.append(osp.join(label_root, 'GT', l))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        img_name = self.image_path[item]

        return pred, gt, img_name

    def __len__(self):
        return len(self.image_path)
        