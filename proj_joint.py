import numpy as np
import cv2

# transfer 3d joint into 2d joint

def proj_joint(joint3d,dx=0.0,dy=0.36,dz=0.0,scalex=1.0,scaley=1.0,width=2.4,height=2.4,zfar=1.2,znear=-1.2,imagew=800, imageh=800):

    joint2d=np.zeros(shape=[joint3d.shape[0],2])


    for x in range(joint3d.shape[0]):

        ScaleMatrix=np.array([[scalex,0.0,0.0,0.0],
                             [0.0,scaley,0.0,0.0],
                             [0.0,0.0,1.0,0.0],
                             [0.0,0.0,0.0,1.0]])

        TransferMatrix=np.array([[1.0,0.0,0.0,dx],
                                [0.0,1.0,0.0,dy],
                                [0.0,0.0,1.0,dz],
                                [0.0,0.0,0.0,1.0]])

        ProjecrMatrix=np.array([[2.0/width,0.0,0.0,0.0],
                               [0.0,2.0/height,0.0,0.0],
                               [0.0,0.0,-2/(zfar-znear),(zfar+znear)/(zfar-znear)],
                               [0.0,0.0,0.0,1.0]])

        this_joint2d =np.matmul(np.matmul(np.matmul(ScaleMatrix,TransferMatrix),ProjecrMatrix),np.transpose([m for m in joint3d[x]]+[1.0]))

        imagex=0.5*imagew*(1+this_joint2d[0])
        imagey = 0.5 * imageh* (1-this_joint2d [1])
        joint2d[x,:]=[imagex,imagey]

    return joint2d







