import torch
import torch.nn as nn
import numpy as np
from src.IoU import *
from src.utils import *
from config import config

CONFIG = config()

def Validate(model, validloader, criterion, valid_loss_min, device, model_path):
    valid_loss = 0
    val_iou = []
    val_losses = []
    model.eval()
    for i, val_data in enumerate(validloader):
        inp, masks, _ = val_data
        inp, masks = inp.to(device), masks.to(device)
        out = model(inp)
        val_target = masks.argmax(1)
        val_loss = criterion(out, val_target.long())
        valid_loss += val_loss.item() * inp.size(0)
        iou = iou_pytorch(out.argmax(1), val_target)
        val_iou.extend(iou)    
    miou = torch.FloatTensor(val_iou).mean()
    valid_loss = valid_loss / len(validloader.dataset)
    val_losses.append(valid_loss)
    print(f'\t\t Validation Loss: {valid_loss:.4f},',f' Validation IoU: {miou:.3f}')
    
    if np.mean(val_losses) <= valid_loss_min:
        torch.save(model.state_dict(), model_path+'/state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min,np.mean(val_losses))+'\n')
        valid_loss_min = np.mean(val_losses)

    return valid_loss, valid_loss_min

def Test_eval(model, testloader, criterion, model_save_pth, device):
      model.load_state_dict(torch.load(model_save_pth))
      model.eval()
      test_loss = 0
      imgs, masks, preds = [], [], []
      for i, test_data in enumerate(testloader):
        img, mask = test_data
        inp, mask = img.to(device), mask.to(device)
        imgs.extend(inp.cpu().numpy())
        masks.extend(mask.cpu().numpy())
        out = model(inp.float())
        preds.extend(out.detach().cpu().numpy())
        target = mask.argmax(1)
        loss = criterion(out, target.long())
        test_loss += loss.item() * inp.size(0)
      test_loss = loss / len(testloader.dataset)
      pred = mask_to_rgb(np.array(preds), CONFIG.id2code)
      print(f"Test loss is: {test_loss:.4f}")
      return np.array(imgs), np.array(masks), np.array(pred)
