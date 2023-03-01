#!/usr/bin/python3
import os
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
from MaskShadowGAN.baseline.models_guided import Generator_S2F

netG_A2B = Generator_S2F(3,3)

    
class MaskShadowGAN_remover:
    def __init__(self,device="cuda:0",pretrained_path=os.path.join("MaskShadowGAN","baseline","netG_A2B.pth")) -> None:
        self.img_transform = transforms.Compose([
        transforms.Resize((400,400), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        self.netG_A2B = Generator_S2F(3,3)
        self.netG_A2B.load_state_dict(torch.load(pretrained_path))
        self.device = device
        self.netG_A2B.to(self.device)
        self.netG_A2B.eval()
        self.to_pil = transforms.ToPILImage()

    def remove_shadow(self,img):
        w, h = img.size
        img_var = (self.img_transform(img).unsqueeze(0)).to(self.device)
        temp_B = self.netG_A2B(img_var)
        fake_B = 0.5*(temp_B.data + 1.0)
        fake_B = np.array(transforms.Resize((h, w))(self.to_pil(fake_B.data.squeeze(0).cpu())))
        return Image.fromarray(fake_B)

    