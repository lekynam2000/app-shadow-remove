#!/usr/bin/python3
import os
import sys
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
from G2R_ShadowRemoval.baseline.model import Generator_S2F, Generator_F2S
from skimage import io, color
from skimage.transform import resize
from G2R_ShadowRemoval.baseline.BDRAR import BDRAR_detector
    
class G2R_remover:
    def __init__(self,device="cuda:0",I_path=os.path.join("G2R_ShadowRemoval","baseline","netG_1.pth"),R_path=os.path.join("G2R_ShadowRemoval","baseline","netG_2.pth")) -> None:
        self.original_transform = transforms.Resize((480,640))
        self.img_transform = transforms.Compose([
        transforms.Resize((400,400), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        self.netG_1 = Generator_S2F()
        self.netG_2 = Generator_F2S()
        self.netG_1.load_state_dict(torch.load(I_path))
        self.netG_2.load_state_dict(torch.load(R_path))
        self.device = device
        self.netG_1.to(self.device)
        self.netG_2.to(self.device)
        self.netG_1.eval()
        self.netG_2.eval()
        self.to_pil = transforms.ToPILImage()

        detector_pretrained = os.path.join("G2R_ShadowRemoval","baseline","BDRAR","ckpt","BDRAR","3000.pth")
        self.detector = BDRAR_detector(pretrained_path=detector_pretrained,device=self.device)

    def remove_shadow(self,image):
        image=self.original_transform(image)
        rgbimage=np.asarray(image)
        labimage = color.rgb2lab(rgbimage)   
        labimage480=resize(labimage,(480,640,3))
        labimage480[:,:,0]=np.asarray(labimage480[:,:,0])/50.0-1.0
        labimage480[:,:,1:]=2.0*(np.asarray(labimage480[:,:,1:])+128.0)/255.0-1.0
        labimage480=torch.from_numpy(labimage480).float()
        labimage480=labimage480.view(480,640,3)
        labimage480=labimage480.transpose(0, 1).transpose(0, 2).contiguous()
        labimage480=labimage480.unsqueeze(0).to(self.device)
        
        
        mask = np.asarray(self.detector.detect_mask(image))
        print(f"mask size: {mask.size}")
        mask480=mask
        mask480=torch.from_numpy(mask480).float()
        mask480=mask480.view(480,640,1)
        mask480=mask480.transpose(0, 1).transpose(0, 2).contiguous()
        mask480=mask480.unsqueeze(0).to(self.device)
        zero = torch.zeros_like(mask480)
        one = torch.ones_like(mask480)
        mask480=torch.where(mask480 > 0.5, one, zero)
        
        real_s480=labimage480.clone()
        real_s480[:,0]=(real_s480[:,0]+1.0)*mask480-1.0
        real_s480[:,1:]=real_s480[:,1:]*mask480

        real_ns480=labimage480.clone()
        real_ns480[:,0]=(real_ns480[:,0]+1.0)*(mask480-1.0)*(-1.0)-1.0
        real_ns480[:,1:]=real_ns480[:,1:]*(mask480-1.0)*(-1.0)
        
        temp_B480 = self.netG_1(real_s480)
        temp_B480 = self.netG_2(temp_B480+real_ns480,mask480*2.0-1.0)

        fake_B480 = temp_B480.data
        fake_B480[:,0]=50.0*(fake_B480[:,0]+1.0)
        fake_B480[:,1:]=255.0*(fake_B480[:,1:]+1.0)/2.0-128.0
        fake_B480=fake_B480.data.squeeze(0).cpu()
        fake_B480=fake_B480.transpose(0, 2).transpose(0, 1).contiguous().numpy()
        fake_B480=resize(fake_B480,(480,640,3))
        fake_B480=color.lab2rgb(fake_B480)
        
        #replace
        mask[mask>0.5]=1
        mask[mask<=0.5]=0
        mask = np.expand_dims(mask, axis=2)
        mask = np.concatenate((mask, mask, mask), axis=-1)
        outputimage=fake_B480*mask+rgbimage*(mask-1.0)*(-1.0)/255.0
        outputimage = (255*outputimage).astype('uint8')
        mask = (255*mask).astype('uint8')
        return {"img":Image.fromarray(outputimage),
                 "mask":Image.fromarray(mask)
                }