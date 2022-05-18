import init
import cv2
import numpy as np
import math
import os
import types

class mask():

    def __init__(self,mask,joint2d,root=None):

        # read mask
        if isinstance(mask,str):
            sourceimg0 = cv2.imread(mask)
            self.sourceimg = np.transpose(sourceimg0, [1, 0, 2])
        else:
            self.sourceimg = np.transpose(mask, [1, 0, 2])

        self.joint2d = joint2d
        if root is not None:
            self.allpartmaskfile = os.path.join(root,'all_body_part.png') # all human body part Mask
            self.LowclothMask= os.path.join(root,'lower_clothing_') # low clothing mask
            self.UpclothMask = os.path.join(root,'upper_clothing_')  # upper clothing mask
            self.DressMask = os.path.join(root,'dress_')  # dress mask

        # the pixel index of foreground
        Maskallindex0 = []
        for i in range(self.sourceimg.shape[0]):
            for n in range(self.sourceimg.shape[1]):
                if (np.sum(self.sourceimg[i, n] - [0.01, 0.01, 0.01]) > 0):
                    Maskallindex0.append([i, n])
        self.Maskallindex = np.array(Maskallindex0)

        # the angle between the big arm to the horizontal axis
        self.lbaang=math.acos(np.dot((self.joint2d[19]-self.joint2d[17]),[-1,0])/np.linalg.norm(self.joint2d[19]-self.joint2d[17]))
        self.rbaang=math.acos(np.dot((self.joint2d[18]-self.joint2d[16]),[1,0])/np.linalg.norm(self.joint2d[18]-self.joint2d[16]))

    def get_allbodypartmask(self,vis=True,iffit=False):

        print("Generate all body parts mask")
        LeftsmallarmMask = partmask(self.joint2d[21], self.joint2d[19], self.Maskallindex, 'lsa')# 左前臂mask
        RightsmallarmMask = partmask(self.joint2d[18], self.joint2d[20], self.Maskallindex, 'rsa')# 右前臂Mask

        # 根据大臂角度 选择执行方案
        if self.lbaang < np.pi * 7/ 16:
            lba = 'LBA'
            bl='BL'
        else:
            lba = 'lba'
            bl = 'bl'
        if self.rbaang < np.pi * 7 / 16:
            rba = 'RBA'
            bl = 'BL'
        else:
            rba = 'rba'
            bl = 'bl'

        LeftbigarmMask = partmask(self.joint2d[19], self.joint2d[17], self.Maskallindex, lba) # 左后臂mask
        RightbigarmMask = partmask(self.joint2d[18], self.joint2d[16], self.Maskallindex, rba) # 右后臂Mask
        BiglegMask = partmask(self.joint2d[0], 0.5 * (self.joint2d[5] + self.joint2d[4]), self.Maskallindex, bl)  # 上腿

        LefthandMask = partmask(self.joint2d[21], self.joint2d[19], self.Maskallindex, 'lh') # 左手
        RighthandMask = partmask(self.joint2d[20], self.joint2d[18], self.Maskallindex, 'rh') # 右手
        NeckMask = partmask(self.joint2d[15], self.joint2d[12], self.Maskallindex, 'neck')  # 脖子
        HeadMask = partmask(self.joint2d[15], self.joint2d[12], self.Maskallindex, 'head') # 头部
        LeftsmalllegMask = partmask(self.joint2d[5], self.joint2d[8], self.Maskallindex, 'lsl') # 左下腿
        RightsmalllegMask = partmask(self.joint2d[4], self.joint2d[7], self.Maskallindex, 'rsl') # 右下腿
        Leftfoot = partmask(self.joint2d[8], self.joint2d[5], self.Maskallindex, 'lf')  # 左脚
        Righfoot = partmask(self.joint2d[7], self.joint2d[4], self.Maskallindex, 'rf') # 右脚

        # all human body part except torso
        mask_except_torso = [LeftsmallarmMask, RightsmallarmMask,LeftbigarmMask,RightbigarmMask,LefthandMask,RighthandMask,
                                  NeckMask,HeadMask,LeftsmalllegMask,RightsmalllegMask, Leftfoot,Righfoot,BiglegMask]
        # generate the torso mask
        bound_box=[]
        if iffit:
            bound_box=[self.joint2d[23],self.joint2d[22],self.joint2d[15]-0.05*(self.joint2d[12]-self.joint2d[15]),self.joint2d[11]]
        else:
            bound_box = [0.5 * (self.joint2d[19] + self.joint2d[17]), 0.5 * (self.joint2d[18] + self.joint2d[16]),
                    self.joint2d[15], self.joint2d[2]]

        mask_shape = [self.sourceimg.shape[0], self.sourceimg.shape[1]]
        mask_torso = get_torso_mask(mask_shape, self.Maskallindex, mask_except_torso, bound_box)

        # all parts mask
        all_parts_mask=mask_except_torso
        all_parts_mask.append(mask_torso)

        # color for vis
        RGB=[]
        for i in range(len(all_parts_mask)):
            RGB.append([75+i*5,200-i*8,0+i*20])

        # vis
        if vis:
            vis_mask(self.sourceimg,all_parts_mask,RGB,self.allpartmaskfile)

        return all_parts_mask

    def get_lowcloth(self,coeff,type):

        '''
        :param w_h: 以裤腰为基准的裤长
        :param type: 下装的类型
        :return:
        '''

        h_w=coeff[0]
        print("Build the upcloth mask:  H/W=", h_w," type=", type)
        assert type=='shorts' or type=='trousers'

        # width constrain
        Tovec1 = [-0.5,0]
        Tovec2 = [0.5,0]

        # initial
        bound1 = self.joint2d[0]
        bound2 = self.joint2d[0]

        # find boundary points
        while 1:
            bound1 = bound1 + Tovec1
            if (list([int(i) for i in np.around(bound1)]) not in list(list(i) for i in self.Maskallindex)):
                break
        while 1:
            bound2 = bound2 + Tovec2
            if (list([int(i) for i in np.around(bound2)]) not in list(list(i) for i in self.Maskallindex)):
                break
        W=np.abs(bound1[0]-bound2[0])

        LowclothMask=[]

        if type=="shorts":
            Hneed=W*h_w
            Htrue=np.abs(self.joint2d[0,1]-0.5*(self.joint2d[2,1]+self.joint2d[1,1]))+\
                  +0.5*np.linalg.norm(self.joint2d[2] - self.joint2d[5])\
                  +0.5*np.linalg.norm(self.joint2d[1] - self.joint2d[4])
            coeff=Hneed/Htrue
            if coeff>1:
                coeff=1
            jointlow=self.joint2d[0]+coeff*(0.5*(self.joint2d[4]+self.joint2d[5])-self.joint2d[0])

            # angle condition
            if self.lbaang < np.pi * 7/ 16:
                bl = 'BL'
            else:
                bl = 'bl'
            lowcloth=partmask(self.joint2d[0], jointlow, self.Maskallindex, bl)
            LowclothMask.append(lowcloth)

        if type=='trousers':
            Hneed = W * h_w-np.abs(self.joint2d[0, 1] - 0.5 * (self.joint2d[2, 1] + self.joint2d[1, 1])) - \
                    - 0.5 * np.linalg.norm(self.joint2d[2] - self.joint2d[5]) \
                    - 0.5 * np.linalg.norm(self.joint2d[1] - self.joint2d[4])
            Htrue = 0.5 * np.linalg.norm(self.joint2d[5] - self.joint2d[8])\
                    + 0.5 * np.linalg.norm(self.joint2d[4] - self.joint2d[7])
            coeff = Hneed / Htrue
            if coeff>0.95:
                coeff=0.95
            jointlow1 = self.joint2d[5] + coeff * (self.joint2d[8] - self.joint2d[5])
            jointlow2 = self.joint2d[4] + coeff * (self.joint2d[7] - self.joint2d[4])
            lowcloth1 = partmask(self.joint2d[5], jointlow1, self.Maskallindex, 'lsl')
            lowcloth2 = partmask(self.joint2d[4], jointlow2, self.Maskallindex, 'rsl')

            if self.lbaang < np.pi * 7/ 16:
                bl = 'BL'
            else:
                bl = 'bl'
            BiglegMask = partmask(self.joint2d[0], 0.5 * (self.joint2d[5] + self.joint2d[4]), self.Maskallindex,bl)  # 上腿

            LowclothMask.append(BiglegMask)
            LowclothMask.append(lowcloth1)
            LowclothMask.append(lowcloth2)

        sourceimgblack=self.sourceimg
        sourceimgblack[:,:]=0
        maskfilename=self.LowclothMask+type+"_"+str(np.round(h_w))+'.png'
        vis_mask(sourceimgblack,LowclothMask,[255,255,255],maskfilename)

        return maskfilename

    def get_upcloth(self, coeff,type,neckneed,returnchoose=None):

        h_w = coeff[0]
        h_l = coeff[1]
        print("Build the upcloth mask:  H/W=", h_w, " H/L=", h_l, " type=", type)
        assert type == 'short' or type == 'long' or type == 'vest'


        UpClothMask = []

        W = np.linalg.norm(self.joint2d[17] - self.joint2d[16])
        Hneed = W * h_w
        Htrue = np.linalg.norm(self.joint2d[0] - 0.5 * self.joint2d[17] - 0.5 * self.joint2d[16])
        Hcoeff = Hneed / Htrue
        jointup = 0.5 * self.joint2d[17] + 0.5 * self.joint2d[16] + Hcoeff * (
                    self.joint2d[0] - 0.5 * self.joint2d[17] - 0.5 * self.joint2d[16])
        if jointup[1] > self.joint2d[2, 1] + 0.5 * (self.joint2d[2, 1] - self.joint2d[0, 1]):
            jointup = 0.5 * self.joint2d[2] + 0.5 * self.joint2d[1] + 0.5 * (
                        0.5 * self.joint2d[2] + 0.5 * self.joint2d[1] - self.joint2d[0])

        if self.lbaang < np.pi * 7 / 16:
            bl = 'BL'
        else:
            bl = 'bl'
        biglegmask = partmask(jointup, 0.5 * (self.joint2d[5] + self.joint2d[4]), self.Maskallindex, bl)

        LeftsmallarmMask = partmask(self.joint2d[21], self.joint2d[19], self.Maskallindex, 'lsa')  # 左臂Mask
        RightsmallarmMask = partmask(self.joint2d[18], self.joint2d[20], self.Maskallindex, 'rsa')  # 右前臂Mask

        if self.lbaang < np.pi * 7 / 16:
            lba = 'LBA'
        else:
            lba = 'lba'
        if self.rbaang < np.pi * 7 / 16:
            rba = 'RBA'
        else:
            rba = 'rba'
        LeftbigarmMask = partmask(self.joint2d[19], self.joint2d[17], self.Maskallindex, lba)  # 左后臂mask
        RightbigarmMask = partmask(self.joint2d[18], self.joint2d[16], self.Maskallindex, rba)  # 右后臂Mask

        LefthandMask = partmask(self.joint2d[21], self.joint2d[19], self.Maskallindex, 'lh')  # 左手
        RighthandMask = partmask(self.joint2d[20], self.joint2d[18], self.Maskallindex, 'rh')  # 右手
        NeckMask = partmask(self.joint2d[15], self.joint2d[12], self.Maskallindex, 'neck')  # 脖子
        HeadMask = partmask(self.joint2d[15], self.joint2d[12], self.Maskallindex, 'head')  # 头部
        LeftsmalllegMask = partmask(self.joint2d[5], self.joint2d[8], self.Maskallindex, 'lsl')  # 左下腿
        RightsmalllegMask = partmask(self.joint2d[4], self.joint2d[7], self.Maskallindex, 'rsl')  # 右下腿
        Leftfoot = partmask(self.joint2d[8], self.joint2d[5], self.Maskallindex, 'lf')  # 左脚
        Righfoot = partmask(self.joint2d[7], self.joint2d[4], self.Maskallindex, 'rf')  # 右脚

        Allapartmaskexceptmain = [LeftsmallarmMask, RightsmallarmMask, LeftbigarmMask, RightbigarmMask, LefthandMask,
                                  RighthandMask,
                                  NeckMask, HeadMask, LeftsmalllegMask, RightsmalllegMask, Leftfoot, Righfoot,
                                  biglegmask]

        BoundBox = [0.5 * (self.joint2d[19] + self.joint2d[17]), 0.5 * (self.joint2d[18] + self.joint2d[16]),
                    self.joint2d[15], 0.5 * (self.joint2d[5] + self.joint2d[2])]
        SourceImgShape = [self.sourceimg.shape[0], self.sourceimg.shape[1]]
        BodyCenter = get_torso_mask(SourceImgShape, self.Maskallindex, Allapartmaskexceptmain, BoundBox)
        UpClothMask.append(BodyCenter)


        if type == 'short':
            Lneed = W * h_l
            Ltrue = 0.5 * (np.linalg.norm(self.joint2d[17] - self.joint2d[19]) + np.linalg.norm(
                self.joint2d[16] - self.joint2d[18]))
            L_coeff = Lneed / Ltrue
            L_coeff = min(L_coeff, 1)
            jointleft = self.joint2d[17] + L_coeff * (self.joint2d[19] - self.joint2d[17])
            jointright = self.joint2d[16] + L_coeff * (self.joint2d[18] - self.joint2d[16])
            LeftbigarmMask = partmask(jointleft, self.joint2d[17], self.Maskallindex, lba)
            RightbigarmMask = partmask(jointright, self.joint2d[16], self.Maskallindex, rba)
            UpClothMask.append(LeftbigarmMask)
            UpClothMask.append(RightbigarmMask)
        if type == 'long':
            if h_l < (0.5 * (np.linalg.norm(self.joint2d[17] - self.joint2d[19]) + np.linalg.norm(
                    self.joint2d[16] - self.joint2d[18])) / W):
                h_l = (0.5 * (np.linalg.norm(self.joint2d[17] - self.joint2d[19]) + np.linalg.norm(
                    self.joint2d[16] - self.joint2d[18])) / W)
            Lneed = W * h_l - 0.5 * (np.linalg.norm(self.joint2d[17] - self.joint2d[19]) +
                                     np.linalg.norm(self.joint2d[16] - self.joint2d[18]))
            Ltrue = 0.5 * (np.linalg.norm(self.joint2d[21] - self.joint2d[19]) +
                           np.linalg.norm(self.joint2d[20] - self.joint2d[18]))
            L_coeff = Lneed / Ltrue
            if L_coeff > 1:
                L_coeff = 1
            jointleft = self.joint2d[19] + L_coeff * (self.joint2d[21] - self.joint2d[19])
            jointright = self.joint2d[18] + L_coeff * (self.joint2d[20] - self.joint2d[18])
            leftsmallarmMask_new = partmask(jointleft, self.joint2d[19], self.Maskallindex, 'lsa')
            rightsmallarmMask_new = partmask(self.joint2d[18], jointright, self.Maskallindex, 'rsa')
            UpClothMask.append(LeftbigarmMask)
            UpClothMask.append(RightbigarmMask)
            UpClothMask.append(leftsmallarmMask_new)
            UpClothMask.append(rightsmallarmMask_new)

        if neckneed:
            NeckMask = partmask(0.5 * self.joint2d[15] + 0.5 * self.joint2d[12], self.joint2d[12], self.Maskallindex,
                                'neck')  # 脖子
            UpClothMask.append(NeckMask)

        rgb = []
        for i in range(len(UpClothMask)):
            rgb.append([255, 255, 255])

        sourceimgblack = self.sourceimg
        sourceimgblack[:, :] = 0
        maskfilename = self.UpclothMask + type + "_" + str(np.round(h_w, 3)) + "_" + str(np.round(h_l, 3)) + '.png'
        vis_mask(sourceimgblack, UpClothMask, rgb, maskfilename)

        returnchoose = True if returnchoose is not None else False  # 全部身体部位分割Mask
        if returnchoose:
            return UpClothMask
        else:
            return maskfilename

    def get_dress(self,coeff,type,neckneed):

        if len(coeff)<3:
            d2_d1=coeff[0]
            h0_d1=coeff[1]
            h1_w = None
            l_w = None
        else:
            d2_d1 = coeff[0]
            h0_d1 = coeff[1]
            h1_w = np.linalg.norm(self.joint2d[0]-0.5*self.joint2d[17]-0.5*self.joint2d[16])/\
                   np.linalg.norm(self.joint2d[17]-self.joint2d[16])
            l_w = coeff[3]

        print("Build the dress mask:  D2_D1=", d2_d1, " H0_D1=", h0_d1, " type=", type)
        assert type == 'skirt' or type == 'shortdress' or 'longdress' or "vestdress"

        Tovec1 = [-0.5,0]
        Tovec2 = [0.5,0]


        bound1 = self.joint2d[0]
        bound2 = self.joint2d[0]

        # 沿方向寻找边界点
        while 1:
            bound1 = bound1 + Tovec1
            if (list([int(i) for i in np.around(bound1)]) not in list(list(i) for i in self.Maskallindex)):
                break
        while 1:
            bound2 = bound2 + Tovec2
            if (list([int(i) for i in np.around(bound2)]) not in list(list(i) for i in self.Maskallindex)):
                break
        bound1=np.round(bound1)
        bound2=np.round(bound2)

        # cal dress coeffcients
        D1=int(np.abs(bound1[0]-bound2[0]))
        H=int(D1*h0_d1)
        D2=int(D1*d2_d1)

        Dresslow = []
        for i in range(D1):
            for n in range(H):
                Dresslow.append([int(bound1[0]+i),int(bound1[1]+n)])

        bound3=[bound1[0]-0.5*(D2-D1),bound1[1]+H]
        boundang=math.acos(np.dot(bound3-bound1, [0, 1]) / np.linalg.norm(bound3-bound1))
        for i in range(int((D2-D1)/2)):
            for n in range(H):
                indexs=[int(bound1[0]-int((D2-D1)/2)+i),int(bound1[1]+n)]
                if math.acos(np.dot((indexs-bound1),[0,1])/np.linalg.norm((bound1-indexs)))<boundang:
                    Dresslow.append(indexs)

        for i in range(int((D2 - D1) / 2)):
            for n in range(int(H)):
                indexs = [int(bound2[0] + i), int(bound2[1] + n)]
                if math.acos(np.dot((indexs - bound2), [0, 1]) / np.linalg.norm((bound2 - indexs))) < boundang:
                    Dresslow.append(indexs)
        Dresslow=np.array(Dresslow)

        Dressall=[]

        if type=='skirt':
            Dressall.append(Dresslow)
            maskfilename = self.DressMask + type + "_" + str(np.round(d2_d1, 3)) + "_" + \
                           str(np.round(h0_d1, 3)) + "_" + '.png'
        if type=='shortdress':
            Dressall.append(Dresslow)
            Dressup=mask.get_upcloth(self,[h1_w,l_w],'short',neckneed,True)
            for i in range(len(Dressup)):
                Dressall.append(Dressup[i])

            maskfilename = self.DressMask + type + "_" + str(np.round(d2_d1, 3)) + "_" + \
                           str(np.round(h0_d1, 3)) + "_" + str(np.round(h1_w, 3)) + \
                           "_" + str(np.round(l_w, 3)) + '.png'

        if type == 'longdress':
            Dressall.append(Dresslow)
            Dressup = mask.get_upcloth(self, [h1_w, l_w], 'long', neckneed,True)
            for i in range(len(Dressup)):
                Dressall.append(Dressup[i])

            maskfilename = self.DressMask + type + "_" + str(np.round(d2_d1, 3)) + "_" + \
                           str(np.round(h0_d1, 3)) + "_" + str(np.round(h1_w, 3)) + \
                           "_" + str(np.round(l_w, 3)) + '.png'

        if type == 'vestdress':
            Dressall.append(Dresslow)
            Dressup = mask.get_upcloth(self, [h1_w, l_w], 'vest', neckneed,True)
            for i in range(len(Dressup)):
                Dressall.append(Dressup[i])

            maskfilename = self.DressMask + type + "_" + str(np.round(d2_d1, 3)) + "_" + \
                           str(np.round(h0_d1, 3)) + "_" + str(np.round(h1_w, 3)) + '.png'

        rgb = []
        for i in range(len(Dressall)):
            rgb.append([255, 255, 255])

        sourceimgblack = self.sourceimg
        sourceimgblack[:, :] = 0
        vis_mask(sourceimgblack, Dressall, rgb, maskfilename)

        return maskfilename


# generate human body part mask based on joints
def partmask(joint1,joint2,maskall,bodypart):

    # select tag based on the human body parts
    if bodypart == 'head' or bodypart == 'rh' or bodypart == 'lh' or bodypart == 'rf' or bodypart == 'lf':
        tag=0
    elif bodypart=='lsl' or bodypart=='rsl':
        tag=1
    elif bodypart=='neck':
        tag=2
    elif bodypart=='bl':
        tag=3
    elif bodypart == 'BL':
        tag = 7
    elif bodypart=='lba' or bodypart=='rba':
        tag=4
    elif bodypart=='LBA' or bodypart=="RBA":
        tag = 6
    elif bodypart=='lsa' or bodypart=='rsa':
        tag = 5

    # joint ray
    base1=np.round(joint2)-np.round(joint1)
    base2=np.round(joint1)-np.round(joint2)

    # width criteria
    Tovec1_1 = [base1[1], -1 * base1[0]] / np.linalg.norm(base1)
    Tovec1_2 = [-1 * base1[1], base1[0]] / np.linalg.norm(base1)
    if (tag == 1 or tag == 5):
        Tovec2_1 = [base2[1], -1 * base2[0]] / np.linalg.norm(base2)
        Tovec2_2 = [-1 * base2[1], base2[0]] / np.linalg.norm(base2)

    # initialize boundary pixels
    bound1_1 = joint1
    bound1_2 = joint1
    if (tag == 1 or tag ==5):
        bound2_1 = joint2
        bound2_2 = joint2

    # find boundary pixels
    while 1:
        bound1_1 = bound1_1 + Tovec1_1
        if (list([int(i) for i in np.around(bound1_1)]) not in list(list(i) for i in maskall)):
            break
    while 1:
        bound1_2 = bound1_2 + Tovec1_2
        if (list([int(i) for i in np.around(bound1_2)]) not in list(list(i) for i in maskall)):
            break

    if (tag == 1 or tag== 5):

        while 1:
            bound2_1 = bound2_1 + Tovec2_1
            if (list([int(i) for i in np.around(bound2_1)]) not in list(list(i) for i in maskall)):
                break
        while 1:
            bound2_2 = bound2_2 + Tovec2_2
            if (list([int(i) for i in np.around(bound2_2)]) not in list(list(i) for i in maskall)):
                break

    # width coefficient
    if tag == 2:
        boundcof = 0.55
    elif tag==3:
        boundcof = 0.57
    elif tag == 4:
        boundcof = 0.50
    elif tag==5 :
        boundcof = 0.52
    elif tag==6:
        boundcof = 0.6
    elif tag==7:
        boundcof = 0.65
    else:
        boundcof=0.95

    # calcualte bound
    if (tag == 1):
        Bound =boundcof* 0.5 * (np.linalg.norm(bound1_1 - bound1_2) + np.linalg.norm(bound2_1 - bound2_2))
    elif (tag==5):
        Bound = boundcof * np.max([np.linalg.norm(bound1_1 - bound1_2),np.linalg.norm(bound2_1 - bound2_2)])
    else:
        Bound = boundcof*np.linalg.norm(bound1_1 - bound1_2)

    Maskpartindex=[]

    for i in range(maskall.shape[0]):

        # calculate theat1 and  theat2
        if (tag==3 or tag == 7 or tag==2):
            a1 = maskall[i] - np.round(joint1)
            thea1 = math.acos(np.dot(a1, [0,1]) / (np.linalg.norm(a1)))
            a2 = maskall[i] - np.round(joint2)
            thea2 = math.acos(np.dot(a2, [0,-1]) / (np.linalg.norm(a2)))

        elif (tag==6):
            a1 = maskall[i] - joint1
            thea1 = math.acos(np.dot(a1, base1) / (np.linalg.norm(a1) * np.linalg.norm(base1)))
            a2 = maskall[i] - np.round(joint2)
            thea2 = math.acos(np.dot(a2, [np.sign(base2[0])*1, 0]) / (np.linalg.norm(a2)))

        else:
            a1=maskall[i]-joint1
            thea1=math.acos(np.dot(a1,base1)/(np.linalg.norm(a1)*np.linalg.norm(base1)))
            a2=maskall[i]-joint2
            thea2=math.acos(np.dot(a2,base2)/(np.linalg.norm(a2)*np.linalg.norm(base2)))

        # traverse all pixels to find the human body part pixels satisfying the orientation and width criteria
        if (tag==0)and thea1 > np.pi /2:

            # distance from pixel to joint ray
            theatobase=np.dot((maskall[i] - joint1), base1)/(np.linalg.norm(base1)*np.linalg.norm(maskall[i] - joint1))
            theatobase=np.sqrt(1-np.square(theatobase))
            disttobase =np.linalg.norm(maskall[i] - joint1)*theatobase
            if disttobase<Bound:
                Maskpartindex.append(maskall[i])

        elif (tag==1 or tag==2 or tag==3 or tag == 4 or tag==5 or tag==6 or tag==7) and thea1<np.pi/2 and thea2<np.pi/2:
            theatobase=np.dot((maskall[i] - joint1), base1)/(np.linalg.norm(base1)*np.linalg.norm(maskall[i] - joint1))
            theatobase=np.sqrt(1-np.square(theatobase))
            disttobase =np.linalg.norm(maskall[i] - joint1)*theatobase
            if disttobase<Bound:
                Maskpartindex.append(maskall[i])

    Maskpartindex=np.array(Maskpartindex)

    return Maskpartindex


# generate the torso mask
def get_torso_mask(Imgshape,Maskallindex,Allapartmaskexceptmain,boundbox):

    # print("Boyd part:bodymain")
    Mediatemask=np.zeros(Imgshape,dtype=int)
    for i in range(Maskallindex.shape[0]):
     Mediatemask[Maskallindex[i,0],Maskallindex[i,1]]=1

    for i in range(len(Allapartmaskexceptmain)):
        for n in range(Allapartmaskexceptmain[i].shape[0]):
            Mediatemask[Allapartmaskexceptmain[i][n, 0],Allapartmaskexceptmain[i][n, 1]]=0
    bodymain=np.argwhere(Mediatemask>0)

    # boundary
    left = boundbox[0][0]
    right = boundbox[1][0]
    up = boundbox[2][1]
    bottom = boundbox[3][1]


    index=[]
    for i in range(len(bodymain)):
        if (bodymain[i, 0] > right or bodymain[i, 0] < left or bodymain[i, 1] < up or bodymain[ i, 1] > bottom):
            index.append(i)
    index=np.unique(index)
    bodymain=np.delete(bodymain,index,axis=0)

    return bodymain


# vis mask
def vis_mask(soureceimg,partmask,rgb,outputfilename):

    for n in range(len(partmask)):
        for i in range(partmask[n].shape[0]):
            soureceimg[partmask[n][i,0],partmask[n][i,1]]=rgb[n]

    soureceimg = np.transpose(soureceimg, [1, 0, 2])
    cv2.imwrite(outputfilename, soureceimg)

