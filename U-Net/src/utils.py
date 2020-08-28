import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torchvision

def imshow(inp, size, title=None):
    '''
        Shows images

        Parameters:
            inp: images
            title: A title for image
    '''
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)

def Color_map(dataframe):
  '''
    Returns the reversed String.

    Parameters:
        dataframe: A Dataframe with rgb values with class maps.

    Returns:
        code2id: A dictionary with color as keys and class id as values.   
        id2code: A dictionary with class id as keys and color as values.
        name2id: A dictionary with class name as keys and class id as values.
        id2name: A dictionary with class id as keys and class name as values.
  '''
  cls = pd.read_csv(dataframe)
  color_code = [tuple(cls.drop("name",axis=1).loc[idx]) for idx in range(len(cls.name))]
  code2id = {v: k for k, v in enumerate(list(color_code))}
  id2code = {k: v for k, v in enumerate(list(color_code))}

  color_name = [cls['name'][idx] for idx in range(len(cls.name))]
  name2id = {v: k for k, v in enumerate(list(color_name))}
  id2name = {k: v for k, v in enumerate(list(color_name))}  
  return code2id, id2code, name2id, id2name

def rgb_to_mask(img, color_map):
    ''' 
        Converts a RGB image mask of shape to Binary Mask of shape [batch_size, classes, h, w]

        Parameters:
            img: A RGB img mask
            color_map: Dictionary representing color mappings

        returns:
            out: A Binary Mask of shape [batch_size, classes, h, w]
    '''
    num_classes = len(color_map)
    shape = img.shape[:2]+(num_classes,)
    out = np.zeros(shape, dtype=np.float64)
    for i, cls in enumerate(color_map):
        out[:,:,i] = np.all(np.array(img).reshape( (-1,3) ) == color_map[i], axis=1).reshape(shape[:2])
    return out.transpose(2,0,1)

def mask_to_rgb(mask, color_map):
    ''' 
        Converts a Binary Mask of shape to RGB image mask of shape [batch_size, h, w, 3]

        Parameters:
            img: A Binary mask
            color_map: Dictionary representing color mappings

        returns:
            out: A RGB mask of shape [batch_size, h, w, 3]
    '''
    single_layer = np.argmax(mask, axis=1)
    output = np.zeros((mask.shape[0],mask.shape[2],mask.shape[3],3))
    for k in color_map.keys():
        output[single_layer==k] = color_map[k]
    return np.uint8(output)

def plotCurves(stats):
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(stats[c], label=c)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training and Validation Loss Curve')
    plt.show()

def Visualize(imgs, title='Original', cols=6, rows=1, plot_size=(16, 16), change_dim=False):
    fig=plt.figure(figsize=plot_size)
    columns = cols
    rows = rows
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.title(title+str(i))
        if change_dim: plt.imshow(imgs.transpose(0,2,3,1)[i])
        else: plt.imshow(imgs[i])
    plt.show()
