import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import sbu_testing_root
from misc import check_mkdir, crf_refine
from model import BDRAR

class BDRAR_detector:
    def __init__(self,pretrained_path="",device="cpu"):
        self.device = device
        self.net = BDRAR().to(self.device)
        self.net.load_state_dict(torch.load(pretrained_path))
        self.net.eval()
        self.img_transform = transforms.Compose([
            transforms.Resize((416,416)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.to_pil = transforms.ToPILImage()
    def detect_mask(self,img):
        w, h = img.size
        img_var = Variable(self.img_transform(img).unsqueeze(0)).to(self.device)
        res = self.net(img_var)
        prediction = np.array(transforms.Resize((h, w))(self.to_pil(res.data.squeeze(0).cpu())))
        prediction = crf_refine(np.array(img.convert('RGB')), prediction)
        return prediction
