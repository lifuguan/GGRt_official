from logging import root
from math import ceil
import numpy as np
from PIL import Image
import os
def concat(root_1,file_id,crop_h,crop_w):
    # images = ['00.png','01.png','02.png','10.png','11.png','12.png','20.png','21.png','22.png']
    img=''
    img_array=''
    row=ceil(378/crop_h)
    col=ceil(504/crop_w)
    images =[]
    for i in range(row):
        for j in range(col):
            images.append(f'{i}{j}.png')
    # pic=['pic_168','pic_008','pic_006','pic_113','pic_419','pic_241']
    # for im in pic:
    root=f'{root_1}/pic_'+file_id[-3:]
    for index,value in enumerate(images):
        image = os.path.join(root,value)
        if index==0:
            img_array = np.array(Image.open(image))
        elif index==1:
            img_array01 = np.array(Image.open(image))
            img_array = np.concatenate((img_array,img_array01),axis=1)#横向拼接
            #img_array = np.concatenate((img_array,img_array2),axis=0)#纵向拼接
        elif index==2:
            img_array02 = np.array(Image.open(image))
            img_array = np.concatenate((img_array,img_array02[:,crop_w*2-496:,:]),axis=1)#横向拼接
        elif index==3:
            img_array1=np.array(Image.open(image))
        elif index==4:
            img_array11 = np.array(Image.open(image))
            img_array1 = np.concatenate((img_array1,img_array11),axis=1)
        elif index==5:
            img_array12 = np.array(Image.open(image))
            img_array1 = np.concatenate((img_array1,img_array12[:,crop_w*2-496:,:]),axis=1)
        elif index==6:
            img_array2=np.array(Image.open(image))[crop_h*2-368:,:,:]
        elif index==7:
            img_array21 = np.array(Image.open(image))[crop_h*2-368:,:,:]
            img_array2 = np.concatenate((img_array2,img_array21),axis=1)
        elif index==8:
            img_array22 = np.array(Image.open(image))[crop_h*2-368:,:,:]
            img_array2 = np.concatenate((img_array2,img_array22[:,crop_w*2-496:,:]),axis=1)
    img_array= np.concatenate((img_array,img_array1,img_array2),axis=0)#横向拼接

    img = Image.fromarray(img_array)

    img.save(f'{root_1}/pic_{file_id[-3:]}/test.png')#图保存为png格式


if __name__ == '__main__':
    root_1=''
    file_id=''
    crop_h=160
    crop_w=224
    concat(root_1,file_id,crop_h,crop_w)