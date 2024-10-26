from __future__ import print_function, division
import sys
sys.path.append('dataloaders')

import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import imageio
from skimage import img_as_ubyte
import os.path as osp

from lib import VideoModel_pvtv2 as Network
from dataloaders import test_dataloader, EvalDataset
from evaluator import Eval_thread
from utils.utils import post_process

def val(test_loader, model):
    
    model.eval()
    with torch.no_grad():
    
        model.cuda(opt.gpu_ids[0])
        
        for i in tqdm(range(test_loader.size)):#
            
            images, images_ycbcr, images_sam, points, gt, name, scene = test_loader.load_data()
            
            gt = np.asarray(gt.squeeze(), np.float32)
            gt /= (gt.max() + 1e-8)
            images = [x.cuda(opt.gpu_ids[0]) for x in images]
            images_ycbcr = [x.cuda(opt.gpu_ids[0]) for x in images_ycbcr]
            images_sam = images_sam[-1].cuda(opt.gpu_ids[0])
            
            pred = model(images, images_ycbcr, images_sam ) 
            
            if name[-5] in ['0','5']:
                
                sam_path = './pred/'+scene +'/'
                if not os.path.exists(sam_path):
                    os.makedirs(sam_path)
                
                out_sam = F.interpolate(pred[-1], size=gt.shape, mode='bilinear', align_corners=False)
                out_sam = torch.where(out_sam > 0.0, torch.ones_like(out_sam), torch.zeros_like(out_sam))
                out_sam = out_sam.data.cpu().numpy().squeeze()
                out_sam = post_process(out_sam)

                imageio.imwrite(sam_path + name.replace('jpg', 'png'), img_as_ubyte(out_sam))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--testsize', type=int, default=352, help='testing dataset size')
    parser.add_argument('--grid', type=int, default=8, help='grid')
    parser.add_argument('--gpu_ids', type=int, default=[0, 1], nargs='+', help='gpu_ids')

    parser.add_argument('--resume', type=str, default='./snapshot/best_checkpoint.pth', help='the best checkpoints')
    parser.add_argument('--dataset',  type=str, default='MoCA')
    parser.add_argument('--test_path', type=str,default='../../dataset/MoCA-Mask/TestDataset_per_sq')
    opt = parser.parse_args() 

    if len(opt.gpu_ids) == 1:
        model = Network(opt).cuda(opt.gpu_ids[0])
    else:
        model = Network(opt).cuda(opt.gpu_ids[0])
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    if os.path.exists(opt.resume):
        params = torch.load(opt.resume)
        model.load_state_dict(params, strict=True)
        print('Loading state dict from: {0}'.format(opt.resume))
    else:
        print("Cannot find model file at {}".format(opt.resume))

    # load data
    print('===> Loading datasets')
    val_loader = test_dataloader(opt)

    val(val_loader, model)
