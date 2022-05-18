import numpy as np
from skimage import io
from skimage import color
from skimage import transform
def imgsynthesis(humanbodyimg,humanbodymask,Targetmask,Clothimg,Clothmask,Boundbox=None):

    # for path
    humanbodyimg_regis=''
    humanbodyimg_rep=''
    humanbodyimg_add=''
    if isinstance(humanbodyimg, str):
        humanbodyimg_regis = humanbodyimg.split('.')[0] + '_regis.png'
        humanbodyimg_rep=humanbodyimg.split('.')[0] + '_rep.png'
        humanbodyimg_add = humanbodyimg.split('.')[0] + '_add.png'
        humanbodyimg=io.imread(humanbodyimg)
    if isinstance(humanbodymask,str):
        humanbodymask=io.imread(humanbodymask)

    '''align virtual human img and its mask'''
    humanbodyimg_gray=color.rgb2gray(humanbodyimg)
    humanbodymask_gray=color.rgb2gray(humanbodymask)

    humanbodyimg_x1=np.min(np.argwhere(humanbodyimg_gray<0.95)[:,1])
    humanbodyimg_x2 = np.max(np.argwhere(humanbodyimg_gray < 0.95)[:, 1])
    humanbodyimg_y1 = np.min(np.argwhere(humanbodyimg_gray < 0.95)[:, 0])
    humanbodyimg_y2 = np.max(np.argwhere(humanbodyimg_gray < 0.95)[:, 0])

    humanbodymask_x1 = np.min(np.argwhere(humanbodymask_gray > 0)[:, 1])
    humanbodymask_x2 = np.max(np.argwhere(humanbodymask_gray > 0)[:, 1])
    humanbodymask_y1 = np.min(np.argwhere(humanbodymask_gray > 0)[:, 0])
    humanbodymask_y2 = np.max(np.argwhere(humanbodymask_gray > 0)[:, 0])

    humanbodyimg_new=np.ones(humanbodymask.shape)
    humanbodyimg_core=transform.resize(humanbodyimg[humanbodyimg_y1:humanbodyimg_y2+1,humanbodyimg_x1:humanbodyimg_x2+1],
                                       (humanbodymask_y2-humanbodymask_y1+1,humanbodymask_x2-humanbodymask_x1+1,3))
    humanbodyimg_new[humanbodymask_y1:humanbodymask_y2+1,humanbodymask_x1:humanbodymask_x2+1]=humanbodyimg_core

    io.imsave(humanbodyimg_regis,humanbodyimg_new)

    for i in range(len(Targetmask)):

        targetmask=None
        if isinstance(Targetmask[i], str):
            targetmask = io.imread(Targetmask[i])
        targetmask_gray = color.rgb2gray(targetmask)

        targetmask_index=np.array(np.argwhere(targetmask_gray >0.1))
        targetmask_index[:,0]=targetmask_index[:,0]+Boundbox[i][0]
        targetmask_index[:, 1] = targetmask_index[:, 1] + Boundbox[i][2]

    io.imsave(humanbodyimg_rep, humanbodyimg_new)

    '''for garment'''
    for i in range(len(Clothimg)):

        clothimg = None
        clothmask = None
        if isinstance(Clothimg[i],str):
            clothimg=io.imread(Clothimg[i])
        if isinstance(Clothmask[i],str):
            clothmask=io.imread(Clothmask[i])
        clothmask_gray = color.rgb2gray(clothmask)

        # 寻找clothmask 的 index
        clothmask_index=np.array(np.argwhere(clothmask_gray >0.1))
        clothmask_index[:,0] = clothmask_index[:,0] + Boundbox[i][0]
        clothmask_index[:, 1] = clothmask_index[:, 1] +  Boundbox[i][2]

        for n in range(clothmask_index.shape[0]):
            humanbodyimg_new[clothmask_index[n, 0], clothmask_index[n, 1], :] = \
                clothimg[clothmask_index[n,0]-Boundbox[i][0],clothmask_index[n,1]-Boundbox[i][2]]/255

    # save end
    io.imsave(humanbodyimg_add,humanbodyimg_new)




