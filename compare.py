from PIL import Image
import numpy as np
import math
TINY_NUMBER = 1e-6 
mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)

def img2mse(x, y, mask=None):
    '''
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    '''

    if mask is None:
        return np.mean((x - y) * (x - y))
    else:
        return np.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (np.sum(mask) * x.shape[-1] + TINY_NUMBER)

def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())

def psnr(img1, img2):
    psnr=img2psnr(img1,img2)
    # mse = numpy.mean( (img1 - img2) ** 2 )
    # if mse == 0:
    #     return 100
    # PIXEL_MAX = 255.0
    return psnr
def compare(root):
    pics=['pic_006','pic_419','pic_008','pic_241','pic_168','pic_113']
    t=0
    i=0
    for pic in pics:
        img1=Image.open(f'{root}/{pic}/test.png')
        img2=Image.open(f'{root}/{pic}/gt.png')
        i=i+1
        i1_array = np.array(img1)
        i2_array = np.array(img2)
        img1 = i1_array.astype(np.float32)
        img2 = i2_array.astype(np.float32)
        img1=img1/255
        img2=img2/255
        img1=(np.clip(img1[None, ...], a_min=0., a_max=1.))
    
        r12=psnr(img1,img2)
        t=t+r12
        print(r12)
    print(f'mean:{t/i}')   