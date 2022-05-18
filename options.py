import numpy as np
import os
class options():
    def __init__(self):

        self.root='example' # dir to your image fold
        self.sub_root=os.listdir(self.root)

        self.prior_file='landmark.txt'
        self.human_body_3d = 'human_body.obj'
        self.human_body_3d_texture = 'human_body_tex.obj'
        self.human_joint='joint.txt'
        self.human_mask='humanbodymask.bmp'
        self.human_mask_dilated='humanbodymask_dilated.bmp'
        self.cloth_mask='cloth_mask.jpg'
        self.cloth_img='cloth.jpg'
        self.grid_img='grid.png'
        self.virtual_human_img='img_syn.png'


        self.gender='female' # your gender: male or female
        self.shapecoeff=[0,0,0,0,0,0,0,0,0,0] # the human body shape of SMPL
        self.neck_include = False # True of false for including neck part in the human body parts mask generating
        self.fitness_kernel = [20, 20] # related to the wearing fitness, two values for two garments
        self.tps_scale=[[20, 10, 6], [32, 16, 18]]  # two lists for two garments and each garments has three tps scale
        self.mask_coeff= [[0.2, 0.2, 0.6], [0.2, 0.2, 0.6]] # the mask fusion coefficient in the corase to fine tps deformaion
        self.ite_num=[4,4] # iterations


