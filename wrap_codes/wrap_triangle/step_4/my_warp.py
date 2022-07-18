'''
    warp image based on arap results
    Basic ideas:
        First warp UV map,
        all the other textures are warped by adapting UV map
'''
import numpy as np
from wrap_triangle.step_4.get_pixelValue import *
import sys

sys.path.append('../wrap_utils/my_cython')
from wrap_triangle.step_4.help_warp import *
import os
import cv2

def get_warpedImage(UV, imgWidth, imgHeight, savePath, target_img):
    '''
        based on UV map and source image, warp image to target image
    '''
    # labelPath = os.path.join(dataPath, 'label')
    # saveLabelPath = os.path.join(savePath, 'label')
    # if not os.path.exists(saveLabelPath):
    #    os.makedirs(saveLabelPath)

    target_img = target_img.astype(np.float) / 255.0
    outImg = textureSampling(target_img, UV)
    outImg = np.reshape(outImg, (imgWidth, imgHeight, -1))

    visImg = my_visualizeImages(outImg, imageType) * mask
    cv2.imwrite(os.path.join(savePath, imageType + '.png'), visImg.astype(np.uint8))

    # gil
    # cv2.imwrite('temp.jpg', (cv2.imread(os.path.join(dataPath, '../../../data/imgHQ00000.png')) * 0.5 + visImg * 0.5))
