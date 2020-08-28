import natsort
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from src.utils import *
from config import config

CONFIG = config()
id2code = CONFIG.id2code

class CamVid_Dataset():
  def __init__(self, img_pth, mask_pth, transform):
    self.img_pth = img_pth
    self.mask_pth = mask_pth
    self.transform = transform
    all_imgs = os.listdir(self.img_pth)
    all_masks = [img_name[:-4] + '_L' + img_name[-4:] for img_name in all_imgs]
    self.total_imgs = natsort.natsorted(all_imgs)
    self.total_masks = natsort.natsorted(all_masks)
  
  def __len__(self):
    return len(self.total_imgs)

  def __getitem__(self, idx):
    img_loc = os.path.join(self.img_pth, self.total_imgs[idx])
    image = Image.open(img_loc).convert("RGB")
    mask_loc = os.path.join(self.mask_pth, self.total_masks[idx])
    mask = Image.open(mask_loc).convert("RGB")
    out_image, rgb_mask = self.transform(image), self.transform(mask)
    out_image = transforms.Compose([transforms.ToTensor()])(out_image) 
    rgb_mask = transforms.Compose([transforms.PILToTensor()])(rgb_mask)
    out_mask = rgb_to_mask(torch.from_numpy(np.array(rgb_mask)).permute(1,2,0), id2code)
    
    return out_image, out_mask, rgb_mask.permute(0,1,2)

class Test():
  def __init__(self, img_pth, mask_pth, transform):
    self.img_pth = img_pth
    self.mask_pth = mask_pth
    self.transform = transform
    all_imgs = os.listdir(self.img_pth)
    all_masks = [img_name[:-4] + '_L' + img_name[-4:] for img_name in all_imgs]
    self.total_imgs = natsort.natsorted(all_imgs)
    self.total_masks = natsort.natsorted(all_masks)
  
  def __len__(self):
    return len(self.total_imgs)

  def __getitem__(self, idx):
    img_loc = os.path.join(self.img_pth, self.total_imgs[idx])
    image = Image.open(img_loc).convert("RGB")
    out_image = self.transform(image)

    mask_loc = os.path.join(self.mask_pth, self.total_masks[idx])
    mask = Image.open(mask_loc).convert("RGB")
    rgb_mask = self.transform(mask)
    
    return out_image, rgb_mask