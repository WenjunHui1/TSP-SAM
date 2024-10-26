import os
import time

import numpy as np
import torch
from torchvision import transforms
import scipy
import scipy.ndimage
from tqdm import tqdm
threshold_sal, upper_sal, lower_sal = 0.5, 1, 0

class Eval_thread():
    def __init__(self, loader, method, dataset, output_dir, cuda=True):
        self.loader = loader
        self.method = method
        self.dataset = dataset
        self.cuda = cuda
        self.logdir=os.path.join(output_dir,'log')
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.curve_cache_dir=os.path.join(output_dir,'curve_cache',dataset,method)
        if not os.path.exists(self.curve_cache_dir):
            os.makedirs(self.curve_cache_dir)
        self.logfile = os.path.join(output_dir, 'result.txt')

    def run(self):
        start_time = time.time()
        
        mae = self.Eval_MAE()
        s_alpha05 = self.Eval_Smeasure(alpha=0.5)
        max_e, mean_e, adp_e = self.Eval_Emeasure()
        meandice, meaniou = self.Eval_DICEIOU()
        
        return s_alpha05, mean_e, mae, meandice, meaniou  
        

    def Eval_MAE(self):
        fLog = open(self.logdir + '/' + self.dataset + '_' + self.method + '_MAE' + '.txt', 'w')
        avg_mae, img_num = 0.0, 0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt, img_id in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                mea = torch.abs(pred - gt).mean()
                if mea == mea:  # for Nan
                    #mae_list.append(mea)
                    avg_mae += mea
                    img_num += 1
                    # print("{} done".format(img_num))
                    fLog.write(img_id + '  ' + str(mea.item()) + '\n')
            avg_mae /= img_num
            fLog.close()
            # print('\n')
            return avg_mae.item()
    
    def Eval_Emeasure(self):
        fLog = open(self.logdir + '/' + self.dataset + '_' + self.method + '_EMeasure' + '.txt', 'w')
        # print('Eval [{:6}] Dataset [Emeasure] with [{}] Method.'.format(self.dataset, self.method))
        avg_e, img_num = 0.0, 0
        adp_e=0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            scores = torch.zeros(255)
            if self.cuda:
                scores = scores.cuda()
            for pred, gt, img_id in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                Q = self._eval_e(pred, gt, 255)
                adp_e+=self._eval_adp_e(pred,gt)
                scores += Q
                img_num += 1
                fLog.write(img_id + '  ' + str(Q.max().item()) + '\n')
                # print("{} done".format(img_num))
            scores /= img_num
            adp_e /= img_num
            for i in range(255):
                fLog.write(str(scores[i].item()) + '\n')
            fLog.close()
            # print('\n')
            return scores.max().item(),scores.mean().item(),adp_e.item()
        
        
    def Eval_DICEIOU(self):
        avg_dice, avg_iou, img_num = 0.0, 0.0, 0
        adp_e=0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            scores_d = torch.zeros(255)
            scores_i = torch.zeros(255)
            if self.cuda:
                scores_d = scores_d.cuda()
                scores_i = scores_i.cuda()
            for pred, gt, img_id in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                score_d, score_i = self._eval_diceiou(pred, gt, 255)
                scores_d += score_d
                scores_i += score_i
                img_num += 1
            scores_d /= img_num
            scores_i /= img_num
            # print('\n')
            return scores_d.mean().item(), scores_i.mean().item()

    def Eval_Smeasure(self, alpha):
        fLog = open(self.logdir + '/' + self.dataset + '_' + self.method + '_SMeasure_' + str(alpha) + '.txt', 'w')
        # print('Eval [{:6}] Dataset [Smeasure] with [{}] Method.'.format(self.dataset, self.method))
        avg_q, img_num = 0.0, 0  # alpha = 0.7; cited from the F-360iSOD
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt, img_id in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                gt[gt >= 0.5] = 1
                gt[gt < 0.5] = 0
                y = gt.mean()
                if y == 0:
                    x = pred.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pred.mean()
                    Q = x
                else:
                    # gt[gt>=0.5] = 1
                    # gt[gt<0.5] = 0
                    Q = alpha * self._S_object(pred, gt) + (1-alpha) * self._S_region(pred, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                img_num += 1
                avg_q += Q.item()
                if np.isnan(avg_q):
                    raise #error

                fLog.write(img_id + '  ' + str(Q.item()) + '\n')
                # print("{} done".format(img_num))
            avg_q /= img_num
            fLog.close()
            # print('\n')
            return avg_q

    def LOG(self, output):
        with open(self.logfile, 'a') as f:
            f.write(output)

    def _eval_diceiou(self, y_pred, y, num):
        if self.cuda:
            score_d = torch.zeros(num).cuda()
            score_i = torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            score_d = torch.zeros(num)
            score_i = torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).float()
            NumRec = torch.where(y_pred_th == 1)[0].shape[-1]
            NumNoRec = torch.where(y_pred_th == 0)[0].shape[-1]
            
            gt = y
            LabelAnd = (y_pred_th + gt) // 2
            NumAnd = torch.where(LabelAnd == 1)[0].shape[-1]
            num_obj = torch.sum(gt)
            num_pred = torch.sum(y_pred_th)
            
            FN = num_obj - NumAnd
            FP = NumRec - NumAnd
            TN = NumNoRec - FN
            
            if NumAnd == 0:
                dice = 0
                iou = 0
            else:
                iou = NumAnd / (FN + NumRec)
                dice = 2 * NumAnd / (num_obj + num_pred);

            score_d[i] = dice
            score_i[i] = iou

        return score_d, score_i



    def _eval_e(self, y_pred, y, num):
        if self.cuda:
            score = torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            score = torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).float()
            if torch.mean(y) == 0.0:  # the ground-truth is totally black
                y_pred_th = torch.mul(y_pred_th, -1)
                enhanced = torch.add(y_pred_th, 1)
            elif torch.mean(y) == 1.0:  # the ground-truth is totally white
                enhanced = y_pred_th
            else:  # normal cases
                fm = y_pred_th - y_pred_th.mean()
                gt = y - y.mean()
                align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
                enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4

            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)

        return score
    def _eval_adp_e(self, y_pred, y):
        th=y_pred.mean() * 2
        y_pred_th=(y_pred >= th).float()
        if torch.mean(y) == 0.0:  # the ground-truth is totally black
            y_pred_th = torch.mul(y_pred_th, -1)
            enhanced = torch.add(y_pred_th, 1)
        elif torch.mean(y) == 1.0:  # the ground-truth is totally white
            enhanced = y_pred_th
        else:  # normal cases
            fm = y_pred_th - y_pred_th.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        return torch.sum(enhanced) / (y.numel() - 1 + 1e-20)


    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

        return prec, recall
    def _eval_adp_f_measure(self,y_pred,y):
        beta2=0.3
        thr=y_pred.mean()*2
        if thr>1:
            thr=1
        y_temp = (y_pred >= thr).float()
        tp = (y_temp * y).sum()
        prec,recall=tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

        adp_f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall+1e-20)
        if torch.isnan(adp_f_score):
            adp_f_score=0.0
        return adp_f_score
    
    def _S_object(self, pred, gt):
        fg = torch.where(gt==0, torch.zeros_like(pred), pred)
        bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1-gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg

        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        if torch.isnan(score):
            raise
        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        # print(Q)

        return Q
    
    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0,cols)).cuda().float()
                j = torch.from_numpy(np.arange(0,rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0,cols)).float()
                j = torch.from_numpy(np.arange(0,rows)).float()
            X = torch.round((gt.sum(dim=0)*i).sum() / total)
            Y = torch.round((gt.sum(dim=1)*j).sum() / total)

        return X.long(), Y.long()
    
    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h*w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3

        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]

        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h*w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
        
        aplha = 4 * x * y *sigma_xy
        beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0

        return Q
