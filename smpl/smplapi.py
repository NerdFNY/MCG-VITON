from smpl.serialization import load_model
import numpy as np
import math
import os



def smplapi(Clothid,Landmark,Root,opt):
    '''
    :param clothid:  garment category
    :param lanmark:  garment landmark
    :param gender:   male or female
    :return:
    '''

    ''' clothid
        1-T shirt  
        2-long sleeve shirt   
        3-short sleeve coat
        4-long sleeve coat
        5-vest
        6-sun-top  
        7-shorts   
        8-pants 
        9-skirt  
        10-short sleeve dress
        11-long sleeve dress
        12-vest dress
        13-sun-top dress
    '''

    m = None

    assert opt.gender=='male' or 'female'
    if opt.gender=='male':
        m = load_model('smpl/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    elif opt.gender=='female':
        m = load_model('smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    for i in range(len(Clothid)):

        # calculate the joint angle based on the clothing landmarks
        clothid=int(Clothid[i])
        landmark=np.array(Landmark[i])
        assert clothid in range(1,14)

        if clothid == 2 or clothid == 11 or clothid == 4:  # 长袖
            # 左大臂
            vector1 = 0.5 * (landmark[3] + landmark[4] - landmark[1] - landmark[0])
            leftangle1 = math.acos(vector1.dot([-1, 0]) / np.linalg.norm(vector1))
            m.pose[53] = leftangle1

            # 左小臂
            vector2 = 0.5 * (landmark[5] + landmark[6] - landmark[3] - landmark[4])
            leftangle2 = math.acos(vector2.dot([-1, 0]) / np.linalg.norm(vector2))
            leftangle2 = leftangle2 - leftangle1
            m.pose[59] = leftangle2

            # 右大臂
            vector3 = 0.5 * (landmark[10] + landmark[11] - landmark[8] - landmark[7])
            rightangle1 = math.acos(vector3.dot([1, 0]) / np.linalg.norm(vector3))
            m.pose[50] = -1 * (rightangle1)

            # 右小臂
            vector4 = 0.5 * (landmark[12] + landmark[13] - landmark[10] - landmark[11])
            rightangle2 = math.acos(vector4.dot([1, 0]) / np.linalg.norm(vector4))
            m.pose[56] = -1 * (rightangle2 - rightangle1)

        elif clothid == 1 or clothid == 10 or clothid == 3:  # 短袖

            # 左大臂
            vector1 = 0.5 * (landmark[2] + landmark[3] - 0.0 * landmark[1] - 1.0 * landmark[0] - landmark[4])
            leftangle1 = math.acos(vector1.dot([-1, 0]) / np.linalg.norm(vector1)) + np.pi / 20
            m.pose[53] = leftangle1

            # 右大臂
            vector3 = 0.5 * (landmark[7] + landmark[8] - 0.0 * landmark[6] - 1.0 * landmark[5] - landmark[9])
            rightangle1 = math.acos(vector3.dot([1, 0]) / np.linalg.norm(vector3)) + np.pi / 20
            m.pose[50] = -1 * rightangle1

        elif clothid == 8:  # 长裤

            # 左大腿
            vector1 = (0.5 * landmark[2] + 0.5 * landmark[3] - landmark[0] - 1 / 3 * (landmark[1] - landmark[0]))
            leftangle1 = math.acos(vector1.dot([0, 1]) / np.linalg.norm(vector1))

            # 右大腿
            vector2 = (0.5 * landmark[4] + 0.5 * landmark[5] - landmark[0] - 2 / 3 * (landmark[1] - landmark[0]))
            rightangle1 = math.acos(vector2.dot([0, 1]) / np.linalg.norm(vector2))

            # 左大腿固有角度
            vector3 = (m.J_transformed[5, 0:-1] - m.J_transformed[2, 0:-1])
            leftangle2 = math.acos(vector3.dot([0, -1]) / np.linalg.norm(vector3))

            # 右大腿固有角度
            vector4 = (m.J_transformed[4, 0:-1] - m.J_transformed[1, 0:-1])
            rightangle2 = math.acos(vector4.dot([0, -1]) / np.linalg.norm(vector4))

            m.pose[8] = -1 * leftangle1
            m.pose[5] = rightangle1

        elif clothid == 7:  # 短裤

            # 左大腿
            vector1 = (0.5 * landmark[2] + 0.5 * landmark[3] - landmark[0] - 1 / 3 * (landmark[1] - landmark[0]))
            leftangle1 = math.acos(vector1.dot([0, 1]) / np.linalg.norm(vector1))
            m.pose[8] = -1 * leftangle1

            # 右大腿
            vector2 = (0.5 * landmark[4] + 0.5 * landmark[5] - landmark[0] - 2 / 3 * (landmark[1] - landmark[0]))
            rightangle1 = math.acos(vector2.dot([0, 1]) / np.linalg.norm(vector2))
            m.pose[5] = rightangle1

        elif clothid == 5 or clothid == 12:  # 背心
            # 左大臂
            m.pose[53] = np.pi / 3
            # 右大臂
            m.pose[50] = -1 * np.pi / 3
        elif clothid == 7:  # 短裙
            m.pose[8] = -1 * np.pi / 10
            m.pose[5] = np.pi / 10


    if len(opt.shapecoeff) != 0:
        for i in range(len(opt.shapecoeff)):
            m.betas[i] = opt.shapecoeff[i]

    for i in range(len(Clothid)):
        objfilename=os.path.join(Root[i],opt.human_body_3d)
        objfilename2 = os.path.join(Root[i], opt.human_body_3d_texture)
        jointfilename=os.path.join(Root[i],opt.human_joint)
        savemodel(objfilename,m)
        savemodel(objfilename2, m,True)
        joint_3d=savejoint(jointfilename,m)

    return joint_3d


# save obj
def savemodel(outmesh_path,m,iftexture=False):
    with open(outmesh_path, 'w') as fp:
        for v in m.r:
            if iftexture:
                fp.write('v %f %f %f 0.638659 0.492244 0.341577\n' % (v[0], v[1], v[2]))
            else:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in m.f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

# save joint
def savejoint(outmesh_path,m):
    Joint=np.zeros(shape=(24,3))
    with open(outmesh_path, 'w') as fp:
        for k in range(24):
            fp.write('%f %f %f \n' % (m.J_transformed[k, 0], m.J_transformed [k, 1], m.J_transformed [k, 2]))
            Joint[k,:]=[m.J_transformed[k, 0], m.J_transformed [k, 1], m.J_transformed [k, 2]]
        fp.close()

    return Joint

