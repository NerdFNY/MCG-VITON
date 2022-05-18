from serialization import load_model
import numpy as np


m=[]
mm = load_model('./basicModel_m_lbs_10_207_0_v1.0.0.pkl')
# m.append(mm)
# # print(mm.J.size)
# # print(mm.J)
# # #print(mm.J[:])
# # outmesh_path = 'E:\Internet+\dataset\SMPL_orignal\smpl\smpl_webuser\Joint\shape.obj'
# # with open( outmesh_path, 'w') as fp:
# #     for v in mm.v_posed:
# #         fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
# #     for f in mm.f+1: # Faces are 1-based, not 0-based in obj files
# #         fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
# # outmesh_path1 = 'E:\Internet+\dataset\SMPL_orignal\smpl\smpl_webuser\Pose\pose.txt'
# # with open(outmesh_path1, 'w') as fp:
# #      for x in range(mm.J.size/3):
# #       fp.write('%d %f %f %f\n'%(x,m[i].pose[x]))