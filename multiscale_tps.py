from skimage import io
from skimage import transform
import numpy as np
from TPS_STN import *
from spatial_gradient import *
import os
from clothing import  *


def mask_merge(maskset,coeff):

    assert np.abs(np.sum(np.array(coeff)))-1<0.1 

    mask_end=np.zeros(maskset[0].shape)

    for i in range(len(maskset)):
        if (np.max(maskset[i])- np.min(maskset[i]))>1.1:
            maskset[i]=maskset[i]/255
        mask_end=mask_end+coeff[i]*maskset[i]

    mask_end=np.round(mask_end)*255

    return mask_end


def multiscale_tps(root,opt, index,human_parts_mask_path,return_choice):


    cloth_mask_path=os.path.join(root,opt.cloth_mask)
    grid_path=os.path.join(root,opt.grid_img)
    cloth_img_path=os.path.join(root,opt.cloth_img)
    tps_scale=opt.tps_scale[index]
    mask_coeff=opt.mask_coeff[index]
    ite_num=opt.ite_num[index]

    assert len(tps_scale)==len(mask_coeff)

    # read img
    cloth_mask = io.imread(cloth_mask_path)
    human_mask = io.imread(human_parts_mask_path)
    cloth_img = io.imread( cloth_img_path)
    grid_img = np.array(io.imread(grid_path))

    # cut the human mask to make it align with the clothing mask
    human_mask_cut,cut_bound_box=mask_cut(human_mask)
    human_mask_cut_path=rename(human_parts_mask_path,'_cut')
    io.imsave(human_mask_cut_path,human_mask_cut)

    # also cut the clothing mask, the purpose is align
    cloth_mask_cut_path=rename(cloth_mask_path,'_cut')
    cloth_img_cut_path = rename(cloth_img_path, '_cut')
    cloth_mask_cut,cloth_img_cut=cloth_cut(cloth_mask,human_mask_cut,cloth_img,cloth_mask_cut_path,cloth_img_cut_path)

    # resize grid img
    grid_img = transform.resize(grid_img, human_mask_cut.shape)

    # record the source size
    cloth_mask_size = list(cloth_mask_cut.shape)
    cloth_img_size=list(cloth_img_cut.shape)
    grid_size=list(grid_img.shape)

    # init warped img
    cloth_mask_warp=cloth_mask_cut
    cloth_img_warp=cloth_img_cut
    grid_warp=grid_img

    # iteration
    for i in range(ite_num):

        print('iteration: ', i)

        cloth_mask_inwarping_set=[]
        grid_width_end,grids_height_end,grid_end=None,None,None

        for n in range(len(tps_scale)):

            if i !=ite_num- 1:
                this_scale=tps_scale[n]
            else:  # the last iter only has one scale
                this_scale=tps_scale[-1]
                
            grid_width = int(human_mask_cut.shape[1] /this_scale)
            grid_height = int(human_mask_cut.shape[0] /this_scale)

            # get the spatial gradient
            grid=gridmove(cloth_mask_warp,human_mask_cut,this_scale)
            grid=np.array(grid.reshape([1, grid_width *grid_height, 2]), dtype=float)

            # tps deformation
            cloth_mask_inwarping= np.array(cloth_mask_warp.reshape([1] + cloth_mask_size + [1]), dtype=float)
            cloth_mask_inwarping= TPS_STN(cloth_mask_inwarping, grid_width, grid_height, grid, cloth_mask_size)
            cloth_mask_inwarping = np.reshape(cloth_mask_inwarping, cloth_mask_size)

            cloth_mask_inwarping_set.append(cloth_mask_inwarping)

            # the last scale of tps deformation
            grid_width_end=grid_width
            grids_height_end=grid_height
            grid_end=np.array(grid)

        # fusion
        cloth_mask_warp=mask_merge(cloth_mask_inwarping_set,mask_coeff) #new cloth_mask_warp

        # apply last scale tps into
        cloth_img_inwarping=np.array(cloth_img_warp.reshape([1] + cloth_img_size), dtype=float)
        cloth_img_inwarping=TPS_STN(cloth_img_inwarping,  grid_width_end, grids_height_end, grid_end, cloth_img_size)
        cloth_img_warp = np.reshape(cloth_img_inwarping,  cloth_img_size) # new cloth_img_Warp

        grid_inwarping=np.array(grid_warp.reshape([1] + grid_size), dtype=float)
        grid_inwarping = TPS_STN(grid_inwarping, grid_width_end,grids_height_end, grid_end, grid_size)
        grid_warp= np.reshape(grid_inwarping, grid_size) # new grid_warp

        # save img
        cloth_mask_process_path="cloth_mask_iter"+str(i)+'.jpg'
        io.imsave(os.path.join(root,cloth_mask_process_path), cloth_mask_warp)

        cloth_img_process_path = "cloth_img_iter" + str(i) + '.jpg'
        io.imsave(os.path.join(root,cloth_img_process_path), cloth_img_warp)

        grid_process_path = "grid_iter" + str(i) + '.jpg'
        io.imsave(os.path.join(root,grid_process_path), grid_warp)

    if return_choice:
        return cloth_mask_warp,cloth_img_warp,cut_bound_box
    else:
        return os.path.join(root,cloth_mask_process_path),os.path.join(root,cloth_img_process_path),human_mask_cut_path,cut_bound_box



def rename(inputfile,newname):
    outputfile=None
    if (inputfile.split('.')[-1]=='png'):
        outputfile=inputfile.replace('.png',newname+'.png')
    elif (inputfile.split('.')[-1]=='jpg'):
        outputfile = inputfile.replace('.jpg',newname +'.jpg')
    return outputfile