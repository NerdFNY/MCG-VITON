import numpy as np
from skimage import io
from skimage import transform
from skimage import color
from scipy import interpolate
from init import *
from scipy.signal import savgol_filter

def gridmove(sourceimg,targetimg,gridscale):

    assert sourceimg.shape==targetimg.shape

    height = int(sourceimg.shape[0] / gridscale)
    width = int(sourceimg.shape[1] / gridscale)
    sourceimg = transform.resize(sourceimg, [height, width])
    targetimg = transform.resize(targetimg, [height, width])

    sourcegray = color.rgb2gray(sourceimg)
    targetgray = color.rgb2gray(targetimg)

    if np.max(sourcegray)-np.min(sourcegray)>1.1:
        sourcegray=np.round(sourcegray/255)
    else:
        sourcegray = np.round(sourcegray)
    if np.max(targetgray)-np.min(targetgray)>1.1:
        targetgray = np.round(targetgray / 255)
    else:
        targetgray = np.round(targetgray)

    gx = 2. / width  # grid x size
    gy = 2. / height  # grid y size
    cx = -1. + gx / 2.  # x coordinate
    cy = -1. + gy / 2.  # y coordinate

    grid= np.empty([height,width, 2], dtype='float32')
    for y in range(height):
      for x in range(width):
        grid[y,x, :] = cx, cy
        cx += gx
      cx = -1. + gx / 2
      cy += gy


    graydiff=np.abs(sourcegray-targetgray)


    Div=np.zeros([graydiff.shape[0]-2,graydiff.shape[1]-2,2])
    gridgap = np.abs(grid[0, 1,0] - grid[0, 0,0])/2

    for y in range(1,graydiff.shape[0]-1):
        for x in range(1,graydiff.shape[1]-1):
            divx = (np.sum(graydiff[y - 1:y + 2,x - 1]) - np.sum(graydiff[y - 1:y + 2,x + 1])) * gridgap / 3
            divy = (np.sum(graydiff[y - 1,x - 1:x + 2]) - np.sum(graydiff[y + 1,x - 1:x + 2])) * gridgap / 3
            Div[y-1,x-1]=[divx,divy]

    Divold=Div

    Div1=np.array(Div)
    Div2=np.array(Div)
    Div3=np.array(Div)
    Div4=np.array(Div)


    if int(np.min([height,width])/3)%2==0:
        windowsize = int(np.min([height, width]) / 3+1)
    else:
        windowsize = int(np.min([height, width]) / 3)

    polyorder=min(3,int(windowsize-1))
    for x in range(Div.shape[0]):
        xnew = savgol_filter(Div[x, :, 0], windowsize, polyorder)
        Div1[x, :, 0] = xnew

        ynew = savgol_filter(Div[x, :, 1], windowsize, polyorder)
        Div2[x, :, 1] = ynew

    for y in range(Div.shape[1]):

        xnew = savgol_filter(Div[:, y, 0], windowsize, polyorder)
        Div3[:, y, 0] = xnew

        ynew = savgol_filter(Div[:, y, 1], windowsize, polyorder)
        Div4[:, y, 1] = ynew

    Divnew=(Div1+ Div2+ Div3+ Div4)/4


    grid[1:-1,1:-1]=grid[1:-1,1:-1]+Divnew
    return grid


def cloth_cut(sourcemask,targetmask,sourceimg,souremaskoutput,sourceimgoutput):

    sourcegray = img_rgbtogray(sourcemask)
    targetgray = img_rgbtogray(targetmask)

    sourcemask_x1=np.min(np.argwhere(sourcegray>0)[:,1])
    sourcemask_x2 = np.max(np.argwhere(sourcegray > 0)[:, 1])
    sourcemask_y1 = np.min(np.argwhere(sourcegray > 0)[:, 0])
    sourcemask_y2 = np.max(np.argwhere(sourcegray > 0)[:, 0])

    targetmask_x1=np.min(np.argwhere(targetgray>0)[:,1])
    targetmask_x2 = np.max(np.argwhere(targetgray> 0)[:, 1])
    targetmask_y1 = np.min(np.argwhere(targetgray > 0)[:, 0])
    targetmask_y2 = np.max(np.argwhere(targetgray > 0)[:, 0])

    sourcemask=sourcemask[sourcemask_y1:sourcemask_y2+1,sourcemask_x1:sourcemask_x2+1]
    sourceimg=sourceimg[sourcemask_y1:sourcemask_y2+1,sourcemask_x1:sourcemask_x2+1]

    sourcemask=transform.resize(sourcemask,(int(targetmask_y2-targetmask_y1),int(targetmask_x2-targetmask_x1)))
    sourceimg=transform.resize(sourceimg,(int(targetmask_y2-targetmask_y1),int(targetmask_x2-targetmask_x1)))

    paddingmask=np.zeros([targetmask.shape[0],targetmask.shape[1],3])
    paddingmask[targetmask_y1: targetmask_y2,targetmask_x1:targetmask_x2,:]= sourcemask

    paddingcloth = np.zeros(targetmask.shape)
    paddingcloth[:,:]=[1,1,1]
    paddingcloth[targetmask_y1: targetmask_y2, targetmask_x1:targetmask_x2,:] = sourceimg

    io.imsave(souremaskoutput, paddingmask)
    io.imsave(sourceimgoutput, paddingcloth)

    return paddingmask,paddingcloth


def img_rgbtogray(img):
    if len(img.shape)==3:
        img=np.mean(img,axis=-1)
        img=np.squeeze(img)
        img=np.array(img)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x,y]<0.1:
                    img[x,y]=int(0)
                else:
                    img[x, y] =int(1)

    return img

