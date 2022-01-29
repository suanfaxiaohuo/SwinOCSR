import os,sys
from PIL import Image
import numpy as np
from tqdm import tqdm_notebook,tqdm
src = './images'
out = './binary_images'
strInFilePath= src
strOutFilePath= out
for i,image in tqdm_notebook(enumerate(os.listdir(strInFilePath))):
    image_path = strInFilePath+image
    imageOutPath = strOutFilePath+image
    img = Image.open(image_path)
    # 模式L为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
    gray = img.convert('L')
    image_file = gray.point(lambda x: 0 if x<200 else 255, '1')
    image_file.save(imageOutPath)

    img_255 = Image.open(imageOutPath)
    img_255 = img_255.convert('L')
    img_255.save(imageOutPath)

