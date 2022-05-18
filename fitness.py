import cv2
import numpy as np
from mask import *

'''dilate overall human body mask'''
def fitness_gloabl(img_path,kernel,save=False,output=None):

    img = cv2.imread(img_path)
    kernel_matrix=np.ones((kernel, kernel))
    img_dilated=cv2.dilate(img, kernel_matrix, iterations=1)

    if save and output!=None:
        cv2.imwrite(outputroot, imgend)

    return img_dilated