from __future__ import annotations
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import random

class ChestXrayPreprocessor:
    def __init__(self,image_size=384,percentile_low=0.5,percentile_high=99.5,training=False,
                 rotation_deg=5.0,translate_frac=0.03,contrast_jitter=0.10):
        self.image_size=image_size; self.p_low=percentile_low; self.p_high=percentile_high
        self.training=training; self.rotation_deg=rotation_deg; self.translate_frac=translate_frac; self.contrast_jitter=contrast_jitter
    def __call__(self,image:Image.Image)->torch.Tensor:
        image=image.convert('L'); arr=np.asarray(image,dtype=np.float32); lo,hi=np.percentile(arr,[self.p_low,self.p_high])
        arr=np.clip(arr,lo,hi); mean,std=float(arr.mean()),float(arr.std()); arr=(arr-mean)/(std+1e-6)
        amin,amax=float(arr.min()),float(arr.max()); u8=((arr-amin)/(amax-amin+1e-6)*255).astype(np.uint8)
        img=Image.fromarray(u8,mode='L'); img=TF.resize(img,[self.image_size,self.image_size],interpolation=InterpolationMode.BILINEAR,antialias=True)
        if self.training:
            angle=random.uniform(-self.rotation_deg,self.rotation_deg); max_shift=int(self.translate_frac*self.image_size)
            translate=[random.randint(-max_shift,max_shift),random.randint(-max_shift,max_shift)]
            img=TF.affine(img,angle=angle,translate=translate,scale=1.0,shear=[0.,0.],interpolation=InterpolationMode.BILINEAR,fill=0)
            img=ImageEnhance.Contrast(img).enhance(1.0+random.uniform(-self.contrast_jitter,self.contrast_jitter))
        x=TF.pil_to_tensor(img).float()/255.0; return (x-x.mean())/(x.std()+1e-6)
