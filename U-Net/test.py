import torch
from torchvision import datasets, transforms
from src.model import UNet
from src.loss import FocalLoss
from config import config
from src.utils import *
from src.Camvid import *
from src.IoU import *
from src.eval import *

CONFIG = config()
path = CONFIG.path
batch = CONFIG.batch
input_size = CONFIG.input_size
load_model_pth = CONFIG.load_model
device = CONFIG.device

if __name__ == "__main__":
    test_transform = transforms.Compose([transforms.Resize(input_size, 0), transforms.ToTensor()])

    #pass transform here-in
    test_data = Test(img_pth = path + 'test/', mask_pth = path + 'test_labels', transform=test_transform)

    #data loaders
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=True)

    model = UNet(3, 32, True).to(device)
    criterion = FocalLoss()

    imgs, masks, pred = Test_eval(model, testloader, criterion, load_model_pth, device)
    print(imgs.shape, masks.shape, pred.shape)

    Visualize(imgs, 'Original Image', 6, 1, change_dim=True)
    Visualize(masks, 'Original Mask', 6, 1, change_dim=True)
    Visualize(pred, 'Predicted mask', 6, 1, change_dim=False)