# -*- coding: utf-8 -*-

"""
# File name:    triangle_wrap_hair.py
# Time :        2021/8/31 11:50
# Author:       xuyang.guo@vipl.ict.ac.cn
# Description:  
"""

from wrap_codes.wrap_triangle.step_3.generate_node import get_node
from wrap_codes.wrap_triangle.step_4.my_loadMesh import load_mesh
from wrap_codes.wrap_triangle.step_4.help_warp import get_warpedUV
import os
from wrap_codes.wrap_triangle.step_4.get_pixelValue import *
from global_value_utils import WRAP_TEMP_FOLDER


def draw_triangle_berkeley_from_arap_obj(file_name, canvas_img, landmarks):
    if file_name.endswith('.obj'):
        vertex, faces, tx_coord = load_mesh(file_name)
        faces -= 1
        vertex = vertex[:, :2]
        vertex = (vertex + 0.5).astype('int')
        vertex = [tuple(vv) for vv in vertex]
    elif file_name.endswith('triangle.txt'):
        vertex, faces = [], []
        with open(file_name, 'r') as f:
            lines = f.readlines()
        ptr = 0
        while lines[ptr][0] == 'v':
            vertex.append(tuple(map(lambda xx: int(float(xx) + 0.5), lines[ptr][:-1].split(' ')[1:])))
            ptr += 1
        for line in lines[ptr:]:
            faces.append(tuple(map(lambda xx: int(xx) - 1, line[:-1].split(' ')[1:])))
    else:
        raise NotImplementedError()
    for face in faces:
            canvas_img = cv2.line(canvas_img, vertex[face[0]], vertex[face[1]], color=(200, 200, 200), thickness=1)
            canvas_img = cv2.line(canvas_img, vertex[face[2]], vertex[face[1]], color=(200, 200, 200), thickness=1)
            canvas_img = cv2.line(canvas_img, vertex[face[2]], vertex[face[0]], color=(200, 200, 200), thickness=1)
    landmarks = landmarks.astype('int')
    for lm in landmarks:
        canvas_img = cv2.circle(canvas_img, tuple(lm), 3, color=(0, 255, 0), thickness=3)
    return canvas_img.astype('uint8')


def get_wrap_UV(landmark_source, landmark_target, wrap_img, warp_temp_folder=None):
    if not warp_temp_folder:
        warp_temp_folder = WRAP_TEMP_FOLDER[0]
    (H, W) = wrap_img.shape[:2]
    if not os.path.exists(warp_temp_folder):
        os.makedirs(warp_temp_folder)
    get_node(landmark_source, landmark_target, W, H, warp_temp_folder, numPoints=50)
    # arap
    triangle_path = os.path.join(warp_temp_folder, 'triangle.txt')
    correspondence_path = os.path.join(warp_temp_folder, 'correspondence.txt')
    wrap_name = os.path.join(warp_temp_folder, 'arap.obj')

    exe_file = os.path.join(os.path.split(__file__)[0], 'libigl_arap/my_arap ')
    os.system('chmod u+x ' + exe_file)
    cmd = exe_file + \
          triangle_path + ' ' + correspondence_path + ' ' \
          + wrap_name + ' ' + str(W) + ' ' + str(H) + ' >/dev/null 2>&1'
    os.system(cmd)

    # wrap
    vertex, faces, tx_coord = load_mesh(wrap_name)
    vertex = np.transpose(vertex, (1, 0))
    faces = np.transpose(faces, (1, 0))
    tx_coord = np.transpose(tx_coord, (1, 0))

    UV, depth_buffer = get_warpedUV(vertex, faces - 1, tx_coord, H, W, c=3)
    # vis_img = (UV * 255.0).astype(np.uint8)
    # cv2.imwrite(os.path.join(sub_folder, 'UV_warp.png'), vis_img)
    UV = UV[:, :, 0:2]

    # gil fix edge
    lin_s = np.linspace(0, 1, UV.shape[0], endpoint=True)
    UV[[0, -1], :, 0] = lin_s
    UV[[0, -1], :, 1] = np.array([[0.0], [1.0 - 1 / UV.shape[0]]])
    UV[-2, :, 1] = np.min(UV[[-2, -1], :, 1], axis=0)
    UV[:, [0, -1], 1] = lin_s[..., None]
    UV[:, [0, -1], 0] = np.array([0.0, 1 - 1 / UV.shape[0]])
    UV[:, -2, 0] = np.min(UV[:, [-2, -1], 0], axis=1)
    UV = np.reshape(UV, (-1, 2))
    return UV, triangle_path, wrap_name


def wrap_by_uv(UV, triangle_path, wrap_name, landmark_source, landmark_target, wrap_img, draw_triangle=False):
    wrap_len = len(wrap_img.shape)
    if wrap_len == 2:
        wrap_img = wrap_img[..., None]
    elif wrap_len != 3:
        raise NotImplementedError()
    (H, W) = wrap_img.shape[:2]

    # get_warpedImage
    wrap_img = wrap_img.astype(np.float)
    out_img = textureSampling(wrap_img, UV)
    out_img = np.reshape(out_img, (W, H, -1))

    # vis_img = np.zeros((H, W, 3))
    # vis_img[:, :, 0] = out_img[:, :, 0]
    # vis_img[:, :, 1] = wrap_img[:, :, 0]
    # cv2.imwrite(os.path.join(WRAP_TEMP_FOLDER, 'wrap_mask.png'), vis_img.astype(np.uint8))
    if draw_triangle:
        skin_color = np.array((0, 85, 255))
        draw_wrap_img = np.tile(wrap_img[..., [0]], [1, 1, 3])
        draw_out_img = np.tile(out_img[..., [0]], [1, 1, 3])
        draw_wrap_img *= skin_color
        draw_out_img *= skin_color
        triangle_img1 = draw_triangle_berkeley_from_arap_obj(triangle_path, draw_wrap_img, landmark_source)
        triangle_img2 = draw_triangle_berkeley_from_arap_obj(wrap_name, draw_out_img, landmark_target)
    else:
        triangle_img1, triangle_img2 = None, None
    if wrap_len == 2:
        out_img = out_img[..., 0]
    return out_img, triangle_img1, triangle_img2


def wrap(landmark_source, landmark_target, wrap_img, draw_triangle=False):
    UV, triangle_path, wrap_name = get_wrap_UV(landmark_source, landmark_target, wrap_img)
    out_img, triangle_img1, triangle_img2 = wrap_by_uv(
        UV, triangle_path, wrap_name, landmark_source, landmark_target, wrap_img, draw_triangle=draw_triangle)
    return out_img, triangle_img1, triangle_img2