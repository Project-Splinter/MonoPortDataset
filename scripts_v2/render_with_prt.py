import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import random
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/'))
import prt_util

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def rotateSH(SH, R):
    SHn = SH
    
    # 1st order
    SHn[1] = R[1,1]*SH[1] - R[1,2]*SH[2] + R[1,0]*SH[3]
    SHn[2] = -R[2,1]*SH[1] + R[2,2]*SH[2] - R[2,0]*SH[3]
    SHn[3] = R[0,1]*SH[1] - R[0,2]*SH[2] + R[0,0]*SH[3]

    # 2nd order
    SHn[4:,0] = rotateBand2(SH[4:,0],R)
    SHn[4:,1] = rotateBand2(SH[4:,1],R)
    SHn[4:,2] = rotateBand2(SH[4:,2],R)

    return SHn

def rotateBand2(x, R):
    s_c3 = 0.94617469575
    s_c4 = -0.31539156525
    s_c5 = 0.54627421529

    s_c_scale = 1.0/0.91529123286551084
    s_c_scale_inv = 0.91529123286551084

    s_rc2 = 1.5853309190550713*s_c_scale
    s_c4_div_c3 = s_c4/s_c3
    s_c4_div_c3_x2 = (s_c4/s_c3)*2.0

    s_scale_dst2 = s_c3 * s_c_scale_inv
    s_scale_dst4 = s_c5 * s_c_scale_inv

    sh0 =  x[3] + x[4] + x[4] - x[1]
    sh1 =  x[0] + s_rc2*x[2] +  x[3] + x[4]
    sh2 =  x[0]
    sh3 = -x[3]
    sh4 = -x[1]
    
    r2x = R[0][0] + R[0][1]
    r2y = R[1][0] + R[1][1]
    r2z = R[2][0] + R[2][1]
    
    r3x = R[0][0] + R[0][2]
    r3y = R[1][0] + R[1][2]
    r3z = R[2][0] + R[2][2]
    
    r4x = R[0][1] + R[0][2]
    r4y = R[1][1] + R[1][2]
    r4z = R[2][1] + R[2][2]
    
    sh0_x = sh0 * R[0][0]
    sh0_y = sh0 * R[1][0]
    d0 = sh0_x * R[1][0]
    d1 = sh0_y * R[2][0]
    d2 = sh0 * (R[2][0] * R[2][0] + s_c4_div_c3)
    d3 = sh0_x * R[2][0]
    d4 = sh0_x * R[0][0] - sh0_y * R[1][0]
    
    sh1_x = sh1 * R[0][2]
    sh1_y = sh1 * R[1][2]
    d0 += sh1_x * R[1][2]
    d1 += sh1_y * R[2][2]
    d2 += sh1 * (R[2][2] * R[2][2] + s_c4_div_c3)
    d3 += sh1_x * R[2][2]
    d4 += sh1_x * R[0][2] - sh1_y * R[1][2]
    
    sh2_x = sh2 * r2x
    sh2_y = sh2 * r2y
    d0 += sh2_x * r2y
    d1 += sh2_y * r2z
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
    d3 += sh2_x * r2z
    d4 += sh2_x * r2x - sh2_y * r2y
    
    sh3_x = sh3 * r3x
    sh3_y = sh3 * r3y
    d0 += sh3_x * r3y
    d1 += sh3_y * r3z
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
    d3 += sh3_x * r3z
    d4 += sh3_x * r3x - sh3_y * r3y
    
    sh4_x = sh4 * r4x
    sh4_y = sh4 * r4y
    d0 += sh4_x * r4y
    d1 += sh4_y * r4z
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
    d3 += sh4_x * r4z
    d4 += sh4_x * r4x - sh4_y * r4y

    dst = x
    dst[0] = d0
    dst[1] = -d1
    dst[2] = d2 * s_scale_dst2
    dst[3] = -d3
    dst[4] = d4 * s_scale_dst4

    return dst

def load_calib(param, render_size=512):
    # pixel unit / world unit
    ortho_ratio = param['ortho_ratio']
    # world unit / model unit
    scale = param['scale']
    # camera center world coordinate
    center = param['center']
    # model rotation
    R = param['R']

    translate = -np.matmul(R, center).reshape(3, 1)
    extrinsic = np.concatenate([R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    # Match camera space to image pixel space
    scale_intrinsic = np.identity(4)
    scale_intrinsic[0, 0] = scale / ortho_ratio
    scale_intrinsic[1, 1] = -scale / ortho_ratio
    scale_intrinsic[2, 2] = scale / ortho_ratio
    # Match image pixel space to image uv space
    uv_intrinsic = np.identity(4)
    uv_intrinsic[0, 0] = 1.0 / float(render_size // 2)
    uv_intrinsic[1, 1] = 1.0 / float(render_size // 2)
    uv_intrinsic[2, 2] = 1.0 / float(render_size // 2)

    intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
    calib = np.concatenate([extrinsic, intrinsic], axis=0)
    return calib


def projection(points, calib):
    return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]


parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--subject', type=str, help='renderppl subject name')
parser.add_argument(
    '-a', '--action', type=str, help='mixamo action name')
parser.add_argument(
    '-f', '--frame', type=int, help='mixamo action frame id')
parser.add_argument(    
    '-o', '--out_dir', type=str, help='output save dir')
args = parser.parse_args()

subject = args.subject
action = args.action
frame = args.frame
save_folder = args.out_dir

mesh_file = os.path.join(
    save_folder, subject, action, f'{frame:06d}', 'mesh.obj')

# calculate prt
prt, face_prt = prt_util.computePRT(mesh_file, 40, 2)

# NOTE: GL context has to be created before any other OpenGL function loads.
size = 512
angl_step = 1 
shs = np.load('./env_sh.npy')

from renderer.gl.init_gl import initialize_GL_context
initialize_GL_context(width=size, height=size, egl=False)

from renderer.gl.prt_render import PRTRender
rndr = PRTRender(width=size, height=size, ms_rate=1, egl=False)
rndr_uv = PRTRender(width=size, height=size, uv_mode=True, egl=False)

from renderer.camera import Camera
from renderer.mesh import load_obj_mesh, compute_tangent
cam = Camera(width=size, height=size)
cam.ortho_ratio = 0.4 * (512 / size)
cam.near = -500
cam.far = 500
cam.sanity_check()

# set path for obj, prt
tex_file = f'../data/renderppl/rigged/{subject}_FBX/tex/{subject}_dif.jpg'

texture_image = cv2.imread(tex_file)
texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(mesh_file, with_normal=True, with_texture=True)
vmin = vertices.min(0)
vmax = vertices.max(0)
up_axis = 1 #if (vmax-vmin).argmax() == 1 else 2

vmed = np.median(vertices, 0)
vmed[up_axis] = 0.5*(vmax[up_axis] + vmin[up_axis])
# y_scale = 180/(vmax[up_axis] - vmin[up_axis])
y_scale = 100

rndr.set_norm_mat(y_scale, vmed)
rndr_uv.set_norm_mat(y_scale, vmed)

tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)    
rndr.set_albedo(texture_image)
rndr_uv.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)   
rndr_uv.set_albedo(texture_image)

for y in tqdm(range(0, 360, angl_step)):
    R = make_rotate(0, math.radians(y), 0)
    if up_axis == 2:
        R = np.matmul(R, make_rotate(math.radians(90),0,0))

    rndr.rot_matrix = R
    rndr_uv.rot_matrix = R
    rndr.set_camera(cam)
    rndr_uv.set_camera(cam)

    # random light
    sh_id = random.randint(0,shs.shape[0]-1)
    sh = shs[sh_id]
    sh_angle = 0.2*np.pi*(random.random()-0.5)
    sh = rotateSH(sh, make_rotate(0, sh_angle, 0).T)

    dic = {'sh': sh, 'ortho_ratio': cam.ortho_ratio, 'scale': y_scale, 'center': vmed, 'R': R}
    calib = load_calib(dic, render_size=size)
    
    rndr.set_sh(sh)        
    rndr.analytic = False
    rndr.use_inverse_depth = False
    rndr.display()

    out_all_f = rndr.get_color(0)
    out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGRA)

    export_calib_file = os.path.join(
        save_folder, subject, action, f'{frame:06d}', 'calib', f'{y:03d}.txt')
    os.makedirs(os.path.dirname(export_calib_file), exist_ok=True)
    # np.save(export_calib_file, dic)
    np.savetxt(export_calib_file, calib)

    export_render_file = os.path.join(
        save_folder, subject, action, f'{frame:06d}', 'render', f'{y:03d}.png')
    os.makedirs(os.path.dirname(export_render_file), exist_ok=True)
    cv2.imwrite(export_render_file, np.uint8(255.0*out_all_f))

    rndr_uv.set_sh(sh)
    rndr_uv.analytic = False
    rndr_uv.use_inverse_depth = False
    rndr_uv.display()

    uv_color = rndr_uv.get_color(0)
    uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)
    
    export_uvrender_file = os.path.join(
        save_folder, subject, action, f'{frame:06d}', 'uv_render', f'{y:03d}.jpg')
    os.makedirs(os.path.dirname(export_uvrender_file), exist_ok=True)
    cv2.imwrite(export_uvrender_file, np.uint8(255.0*uv_color))

    if uv_color.sum() == 0:
        break
