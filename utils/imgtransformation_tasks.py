#Adapted from https://github.com/VITA-Group/Adv-SS-Pretraining
import torch
import numpy as np

### for images 1x512x512 and 16 sections_jigsaw ((512/128)**2=16): 512/128 = sqrt(16); IMG_SIZE = N * sqrt(NB_JIGSAW)
## N*num = 512

def split_image(image, N=128):
    """
    image: (C, W, H)
    """
    batches = []

    for i in list(torch.split(image, N, dim=1)):
        batches.extend(list(torch.split(i, N, dim=2)))

    return batches

def concat(batches, num=4):
    """
    batches: [(C,W1,H1)]
    """
    batches_list=[]

    for j in range(len(batches) // num):
        batches_list.append(torch.cat(batches[num*j:num*(j+1)], dim=2))

    concatenated =  torch.cat(batches_list,dim=1)
    return concatenated

def rotation(input):
    batch = input.shape[0]
    target = torch.tensor(np.random.permutation([0,1,2,3] * (int(batch / 4) + 1)), device = input.device)[:batch]
    target = target.long()
    image = torch.zeros_like(input)
    image.copy_(input)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(input[i, :, :, :], target[i], [1, 2])

    return image, target

def jigsaw(input, permutation, bias=0.9):
    cur_batch_size = input.size(0)
    img_size = input.size(-1)
    N = int(img_size / np.sqrt(permutation.shape[1]))

    jig_input = torch.zeros_like(input)
    jig_target = torch.zeros(cur_batch_size)

    for idx in range(cur_batch_size):
        img = input[idx, :]
        batches = split_image(img, N=N)
        order = np.random.randint(len(permutation) + 1)

        rate = np.random.rand()
        if rate > bias:
            order = 0

        if order == 0:
            new_batches = batches
        else:
            new_batches = [batches[permutation[order - 1][j]] for j in range(len(permutation[0]))]

        concatenated= concat(new_batches, img_size//N)
        jig_input[idx] = concatenated
        jig_target[idx] = order

    return jig_input, jig_target