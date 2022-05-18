import os
from smpl.smplapi import *
from img_syn import  *
from options import *
from clothing import *
from proj_joint import *
from fitness import *
from multiscale_tps import *

def main():

    opt=options()
    cloth_category_set=[]
    landmark_set=[]
    cloth_img_root_set=[]
    human_body_mask_cut_path_set=[]
    cut_bound_box_set=[]
    cloth_mask_warp_path_set=[]
    cloth_img_warp_path_set=[]

    for i in range(len(opt.sub_root)):
        print(opt.sub_root)
        print(i)

        cloth_img_root=os.path.join(opt.root,opt.sub_root[i])
        cloth_prior_path=os.path.join(cloth_img_root,opt.prior_file)

        '''step 1 read clothing prior from the txt file'''
        cloth_category, cloth_landmark = cloth_prior_read(cloth_prior_path)

        cloth_category_set.append(cloth_category)
        landmark_set.append(cloth_landmark)
        cloth_img_root_set.append(cloth_img_root)

        '''step 2 generate 3d human body model based on the clothing pose '''
        human_joint_3d=smplapi(cloth_category_set, landmark_set,cloth_img_root_set,opt)
        print("Note: 3d human body done!, render it and continue")
        os.system("pause")

        '''step 3 project 3d human joints into 2d joints'''
        human_joint_2d = proj_joint(human_joint_3d)
        human_mask_path=os.path.join(cloth_img_root_set[i], opt.human_mask)

        '''step 4 adjust the wearing fitness'''
        human_mask_dilated = fitness_gloabl(human_mask_path, opt.fitness_kernel[i])

        '''step 5 calculate the clothing coefficients'''
        cloth_coeff= cal_cloth_coeff(cloth_category, cloth_landmark)

        '''step 5.5 optional visualize the human mask'''
        human_mask=mask(human_mask_dilated,human_joint_2d,cloth_img_root)
        # human_mask.get_allbodypartmask() # vis mask

        '''step 6 generate human body parts mask '''
        human_parts_mask_path=None
        # upper clothing
        if cloth_category in [1,2,3,4,5,6]:
            if cloth_category in [1,3,5,6]:
                subtype='short'
            else:
                subtype='long'
            human_parts_mask_path=human_mask.get_upcloth(cloth_coeff,subtype,opt.neck_include)
        # lower clothing
        elif cloth_category in [7,8]:
            if cloth_category==7:
                subtype = 'shorts'
            else:
                subtype = 'trousers'
            human_parts_mask_path=human_mask.get_lowcloth(cloth_coeff, subtype)
        # dress
        elif cloth_category in [9,10,11,12,13]:
            if cloth_category == 9:
                subtype = 'skirt'
            elif cloth_category == 11:
                subtype = 'longdress'
            else:
                subtype = 'shortdress'
            human_parts_mask_path=human_mask.get_dress(cloth_coeff, subtype)

        '''step 7 the coare to fine tps deformation '''

        cloth_mask_warp_path,  cloth_img_warp_path,  human_body_mask_cut_path, cut_bound_box=multiscale_tps(cloth_img_root,opt,i,human_parts_mask_path,False)
        human_body_mask_cut_path_set.append(human_body_mask_cut_path)
        cut_bound_box_set.append(cut_bound_box)
        cloth_img_warp_path_set.append(cloth_img_warp_path)
        cloth_mask_warp_path_set.append(cloth_mask_warp_path)

    '''step 8 img synthesis'''
    imgsynthesis(os.path.join(cloth_img_root_set[-1], opt.virtual_human_img), os.path.join(cloth_img_root_set[-1], opt.human_mask),
                 human_body_mask_cut_path_set, cloth_img_warp_path_set, cloth_mask_warp_path_set, cut_bound_box_set)

if __name__=="__main__":
  main()