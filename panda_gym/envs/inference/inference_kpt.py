import argparse
import cv2
import os
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from panda_gym.envs.inference.models.model_clip import CLIPLingUNet
from torchvision import transforms

class KptInference:
    def __init__(self, checkpoint_start='checkpoint_start/model.pth', checkpoint_end='checkpoint_end/model.pth', ROOT_DIR='/host/panda_gym/envs/inference'):
        sys.path.insert(0, ROOT_DIR)
        sys.path.append(os.path.join(ROOT_DIR, 'models'))
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        cfg = {'train': {'batchnorm': True, 'lang_fusion_type': 'mult'}}
        self.img_dim = 480
        self.start_model = CLIPLingUNet((self.img_dim, self.img_dim, 3), 1, cfg, 'cuda:0', None)
        #self.end_model = CLIPLingUNet((self.img_dim, self.img_dim, 4), 1, cfg, 'cuda:0', None)

        start_checkpoint_path = '%s/%s'%(ROOT_DIR, checkpoint_start)
        self.start_model.load_state_dict(torch.load(start_checkpoint_path))
        
        end_checkpoint_path = '%s/%s'%(ROOT_DIR, checkpoint_end)
        #self.end_model.load_state_dict(torch.load(end_checkpoint_path))

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
            self.start_model = self.start_model.cuda()
            #self.end_model = self.end_model.cuda()

        self.start_model.eval()
        #self.end_model.eval()

        self.transform = transforms.Compose([transforms.ToTensor()])

    def normalize(self, x):
        return F.normalize(x, p=1)
    
    def gauss_2d_batch(self, width, height, sigma, U, V, normalize_dist=False):
        U.unsqueeze_(1).unsqueeze_(2)
        V.unsqueeze_(1).unsqueeze_(2)
        X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
        X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
        G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
        if normalize_dist:
            return self.normalize(G).double()
        return G.double()

    def run_inference(self, img_np, text, kpt=None, save_path=None):
        img_t = self.transform(img_np)
        if kpt is not None:
            kpt = np.reshape(kpt, (-1,2))
            U = torch.from_numpy(kpt[:,0]).cuda()
            V = torch.from_numpy(kpt[:,1]).cuda()
            gaussians = self.gauss_2d_batch(self.img_dim, self.img_dim, 6, U, V)
            start_gauss = gaussians[0].unsqueeze(0)
            img_t = torch.vstack((img_t.cuda(), start_gauss))
        img_t = img_t.unsqueeze(0).float()

        if kpt is None:
            heatmap = self.start_model(img_t.cuda(), text)
        else:
            print(img_t.shape)
            heatmap = self.end_model(img_t.cuda(), text)

        heatmap = heatmap.detach().cpu().numpy()[0]

        h = heatmap[0]
        pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
        vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.5, vis, 0.5, 0)
        overlay = cv2.circle(overlay, (pred_x,pred_y), 4, (0,0,0), -1)
        result = np.hstack((img_np, vis, overlay))
        cv2.putText(result, text, (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2, 2)
        cv2.imshow('img', result)
        cv2.waitKey(0)
        if save_path is not None:
            cv2.imwrite(save_path, result)

        return np.array([pred_x, pred_y])
