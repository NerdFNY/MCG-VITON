import collections
import numpy as np
from skimage import io
import json
import matplotlib.pyplot as plt
from skimage import color
from mpl_toolkits.mplot3d import Axes3D

'''
read clothing prior from the txt file
and the txt files structurre is as:
=======================
clothing category index
landmark1_x landmark2_x
...
landmarkn_x landmarkn_x
=======================
'''

'''read garment prior'''
def cloth_prior_read(filename):
    with open(filename) as file:
        info = []
        while 1:
            line = file.readline()
            if not line:
                break
            info.append(line)
    file.close()
    cloth_cate=int(info[0])
    landmarks=[[int(str.split(' ')[0]),int(str.split(' ')[1])] for str in info[1:]]
    return cloth_cate,landmarks

'''calculate garment coefficients based on landmark'''
def cal_cloth_coeff(clothid,landmark):

    landmark=np.array(landmark)
    coeff=[]
    if clothid ==2:
        base=np.linalg.norm(landmark[0]-landmark[7])
        h=0.5*(np.abs(0.5*landmark[0,1]+0.5*landmark[2,1]-landmark[15,1])+np.abs(0.5*landmark[7,1]+0.5*landmark[9,1]-landmark[14,1]))
        coeff.append(h/base)
        l=0.5*(np.linalg.norm(0.5*landmark[0]+0.5*landmark[2]-landmark[6])+np.linalg.norm(0.5*landmark[7]+0.5*landmark[9]-landmark[13]))
        coeff.append(l / base)

    elif clothid ==1:
        base = np.linalg.norm(landmark[0] - landmark[5])
        h = 0.5 * (np.abs(landmark[0, 1] - landmark[10, 1]) + np.abs(landmark[5, 1] - landmark[11, 1]))
        coeff.append(h / base)
        l = 0.5 * (np.linalg.norm(0.5 * landmark[0] + 0.5 * landmark[1] - landmark[2]) + np.linalg.norm(
            0.5 * landmark[5] + 0.5 * landmark[6] - landmark[7]))
        coeff.append(l / base)

    elif clothid==5:
        base = np.linalg.norm(landmark[0] - landmark[1])
        h = 0.5 * (np.abs(landmark[0, 1] - landmark[2, 1]) + np.abs(landmark[1, 1] - landmark[3, 1]))
        coeff.append(h / base)
        l = 2.0
        coeff.append(0.5)

    elif clothid==8 or clothid==7:
        base=np.linalg.norm(landmark[0] - landmark[1])
        h=0.5*np.linalg.norm(landmark[0]-landmark[2])+0.5*np.linalg.norm(landmark[1]-landmark[5])
        coeff.append(h/base)

    elif clothid==9:
        d2_d1 = np.linalg.norm(landmark[2]-landmark[3])/np.linalg.norm(landmark[0]-landmark[1])
        h0_d1 = np.abs(0.5*landmark[2,1]+0.5*landmark[3,1]-0.5*landmark[0,1]-0.5*landmark[1,1])\
                /np.linalg.norm(landmark[0]-landmark[1])
        coeff.append(d2_d1)
        coeff.append(h0_d1)

    elif clothid==11:
        d2_d1 = np.linalg.norm(landmark[16] - landmark[17]) / np.linalg.norm(landmark[15] - landmark[14])
        h0_d1 = np.abs(0.5 * landmark[15, 1] + 0.5 * landmark[14, 1] - 0.5 * landmark[16, 1] - 0.5 * landmark[17, 1]) \
                / np.linalg.norm(landmark[15] - landmark[14])
        base = np.linalg.norm(landmark[0] - landmark[7])
        h = 0.5 * (np.abs(0.5 * landmark[0, 1] + 0.5 * landmark[2, 1] - landmark[15, 1]) + np.abs(
            0.5 * landmark[7, 1] + 0.5 * landmark[9, 1] - landmark[14, 1]))
        l = 0.5 * (np.linalg.norm(0.5 * landmark[0] + 0.5 * landmark[2] - landmark[6]) + np.linalg.norm(
            0.5 * landmark[7] + 0.5 * landmark[9] - landmark[13]))

        coeff.append(d2_d1)
        coeff.append(h0_d1)
        coeff.append(h/base)
        coeff.append(l / base)

    elif clothid==10:
        d2_d1 = np.linalg.norm(landmark[12] - landmark[13]) / np.linalg.norm(landmark[10] - landmark[11])
        h0_d1 = np.abs(0.5 * landmark[10, 1] + 0.5 * landmark[11, 1] - 0.5 * landmark[12, 1] - 0.5 * landmark[13, 1]) \
                / np.linalg.norm(landmark[10] - landmark[11])
        base = np.linalg.norm(landmark[0] - landmark[5])
        h = 0.5 * (np.abs(landmark[0, 1] - landmark[10, 1]) + np.abs(landmark[5, 1] - landmark[11, 1]))
        l = 0.5 * (np.linalg.norm(0.5 * landmark[0] + 0.5 * landmark[1] - landmark[2]) + np.linalg.norm(
            0.5 * landmark[5] + 0.5 * landmark[6] - landmark[7]))
        coeff.append(d2_d1)
        coeff.append(h0_d1)
        coeff.append(h / base)
        coeff.append(l / base)

    elif clothid==12:
        d2_d1 = np.linalg.norm(landmark[4] - landmark[5]) / np.linalg.norm(landmark[2] - landmark[3])
        h0_d1 = np.abs(0.5 * landmark[2, 1] + 0.5 * landmark[3, 1] - 0.5 * landmark[4, 1] - 0.5 * landmark[5, 1]) \
                / np.linalg.norm(landmark[2] - landmark[3])
        base = np.linalg.norm(landmark[0] - landmark[1])
        h = 0.5 * (np.abs(landmark[0, 1] - landmark[1, 1]) + np.abs(landmark[2, 1] - landmark[3, 1]))

        coeff.append(d2_d1)
        coeff.append(h0_d1)
        coeff.append(h / base)
        coeff.append(0.5)

    return coeff


def mask_cut(mask):

    # transfer into gray
    maskgray=color.rgb2gray(np.array(mask))/255

    # bound box
    mask_x1 = np.min(np.argwhere(maskgray > 0)[:, 1])
    mask_x2 = np.max(np.argwhere(maskgray > 0)[:, 1])
    mask_y1 = np.min(np.argwhere(maskgray > 0)[:, 0])
    mask_y2 = np.max(np.argwhere(maskgray > 0)[:, 0])

    # padding
    coeff=1.2
    xpadding=max(10,(coeff-1.0) * (mask_x2-mask_x1) / 2)  # x 轴方向增减的数量
    ypadding = max(10, (coeff - 1.0) * (mask_y2 - mask_y1) / 2)  # y 轴方向增减的数量

    mask_x1_new =int(mask_x1 - xpadding)
    mask_x2_new = int(mask_x2 + xpadding)
    mask_y1_new=int(mask_y1- ypadding)
    mask_y2_new = int(mask_y2 + ypadding)
    masknew=np.array(mask[mask_y1_new:mask_y2_new+1,mask_x1_new:mask_x2_new+1,:])

    return masknew,[mask_y1_new,mask_y2_new,mask_x1_new,mask_x2_new]


