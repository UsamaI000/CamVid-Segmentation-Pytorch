import torch
import torch.optim as optim
from torchvision import datasets, transforms
from src.model import *
from src.loss import FocalLoss
from config import config
from src.utils import *
from src.Camvid import *
from src.IoU import *
from src.eval import *

CONFIG = config()

path = CONFIG.path
batch = CONFIG.batch
lr = CONFIG.lr
epochs = CONFIG.epochs
device = CONFIG.device
print(f"The device being used is: {device}\n")
id2code = CONFIG.id2code
input_size = CONFIG.input_size
model_sv_pth = CONFIG.model_path
load_model_pth = CONFIG.load_model

def train(model, trainloader, validloader, criterion, optimizer, epochs, device, load_pth, model_sv_pth, plot=True, visualize=False, load_model=False):
    if load_model: model.load_state_dict(torch.load(load_pth))
    model.train()
    stats = []
    valid_loss_min = np.Inf
    print('Training Started.....')
    for epoch in range(epochs):
        train_loss = 0
        train_iou = []
        for i, data in enumerate(trainloader):
            inputs, mask, rgb = data
            inputs, mask = inputs.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(inputs.float())
            target = mask.argmax(1)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0) 
            iou = iou_pytorch(output.argmax(1), target)
            train_iou.extend(iou)     
            if visualize and epoch%10==0 and i == 0:
                print('The training images')
                show_databatch(inputs.detach().cpu(), size=(8,8))
                print('The original masks')
                show_databatch(rgb.detach().cpu(), size=(8,8))
                RGB_mask =  mask_to_rgb(output.detach().cpu(), id2code)
                print('Predicted masks')
                show_databatch(torch.tensor(RGB_mask).permute(0,3,1,2), size=(8,8))
        miou = torch.FloatTensor(train_iou).mean()
        train_loss = train_loss / len(trainloader.dataset)
        print('Epoch',epoch,':',f'Lr ({optimizer.param_groups[0]["lr"]})',f'\n\t\t Training Loss: {train_loss:.4f},',f' Training IoU: {miou:.3f},')
        
        with torch.no_grad():
            valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
            
        stats.append([train_loss, valid_loss])
        stat = pd.DataFrame(stats, columns=['train_loss', 'valid_loss'])

    print('Finished Training')
    if plot: plotCurves(stat)


if __name__ == "__main__":
    
    #Define transforms for the training data and validation data
    train_transforms = transforms.Compose([transforms.Resize(input_size, 0)])
    valid_transforms = transforms.Compose([transforms.Resize(input_size, 0)])

    #pass transform here-in
    train_data = CamVid_Dataset(img_pth = path + 'train/', mask_pth = path + 'train_labels/', transform = train_transforms)
    valid_data = CamVid_Dataset(img_pth = path + 'val/', mask_pth = path + 'val_labels/', transform = valid_transforms)

    #data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch, shuffle=True)

    model = UNet(3, 32, True).to(device)
    criterion = FocalLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)

    train(model, trainloader, validloader, criterion, optimizer, epochs, device, load_model_pth, model_sv_pth, plot=True, visualize=True, load_model=False)
