import skimage.io as io
from skimage.transform import  rescale,resize
from skimage.util import img_as_uint,img_as_ubyte
from skimage.color import rgb2gray
from skimage import exposure
import os
import numpy as np

def processing_image(filename):
    img=io.imread(filename)
    if img.ndim != 2:
        img_gray = rgb2gray(img[..., :3])
    else:
        img_gray = img #image is already greyscale
    img_resized = resize(img_gray, (512, 512))#convert image size to 512*512
    img_rescaled=(img_resized-np.min(img_resized))/(np.max(img_resized)-np.min(img_resized))#min-max normalization 
    img_enhanced=exposure.equalize_adapthist(img_rescaled)#adapt hist
    img_resized_8bit=img_as_ubyte(img_enhanced)
    
    return img_resized_8bit