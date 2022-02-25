import numpy as np
from skimage import color
import torch

def semseg_single_image( predicted, img):

    if isinstance(img, torch.Tensor) and img.shape[0]==3:
        img = torch.transpose(torch.transpose(img, 0, 1), 1,2).cpu().detach().numpy()
        predicted = torch.transpose(torch.transpose(predicted, 0, 1), 1,2).cpu().detach().numpy()
    label = np.argmax(predicted, axis=-1)
    COLORS = ('white','red', 'blue', 'yellow', 'magenta',
            'green', 'indigo', 'darkorange', 'cyan', 'pink',
            'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
            'purple', 'darkviolet')
    rgb = (img + 1.) / 2.

    pred = color.label2rgb(np.squeeze(label), np.squeeze(rgb), colors=COLORS, kind='overlay')[np.newaxis,:,:,:]
    #preds = [color.label2rgb(np.squeeze(x), np.squeeze(y), colors=COLORS, kind='overlay')[np.newaxis,:,:,:] for x,y in zip(label, rgb)]
    merged = pred.squeeze()

    return np.uint8(merged*256)
