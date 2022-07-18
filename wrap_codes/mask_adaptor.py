import numpy as np
import os
import cv2
import pickle as pkl
from util.imutil import write_rgb
from wrap_codes.wrap_triangle import triangle_wrap_hair
from global_value_utils import PARSING_COLOR_LIST, HAIR_IDX, GLOBAL_DATA_ROOT
from external_code.face_parsing.my_parsing_util import FaceParsing
from external_code.landmarks_util import predictor_dict, detector

LANDMARK_81_DIC = None


def load_landmark_dic():
    global LANDMARK_68_DIC, LANDMARK_81_DIC
    with open(os.path.join(GLOBAL_DATA_ROOT, 'landmark81.pkl'), 'rb') as f:
        LANDMARK_81_DIC = pkl.load(f)


def get_parsing_label(data_dir, file_name):
    # wh = 512
    # img = np.zeros((wh, wh), dtype=np.uint8)
    # parsing_dir = os.path.join(data_dir, 'parsing')
    # for label_idx, label_name in enumerate(PARSING_LABEL_LIST):
    #     label_file = file_name[:-4] + '_' + label_name + '.png'
    #     label_file = os.path.join(parsing_dir, label_file)
    #     if os.path.exists(label_file):
    #         mask = cv2.cvtColor(cv2.imread(label_file), cv2.COLOR_BGR2GRAY)
    #         index = np.where(mask == 255)
    #         img[index[0], index[1]] = label_idx
    #     elif label_idx == 1 and label_name == 'skin_other':
    #         label_file = file_name[:-4] + '_' + 'skin' + '.png'
    #         label_file = os.path.join(parsing_dir, label_file)
    #         if os.path.exists(label_file):
    #             mask = cv2.cvtColor(cv2.imread(label_file), cv2.COLOR_BGR2GRAY)
    #             index = np.where(mask == 255)
    #             img[index[0], index[1]] = label_idx
    img = cv2.imread(os.path.join(data_dir, 'label', file_name), cv2.IMREAD_GRAYSCALE)
    return img


def get_parsing_show(img, fuse_img=None):
    """
    :param fuse_img:
    :param img:
    :return: RGB
    """
    if len(img.shape) == 3:
        img = img[:, :, 0]
    result = np.zeros((*img.shape, 3), dtype=np.uint8)
    for label_idx in np.unique(img):
        index = np.where(img == label_idx)
        result[index[0], index[1]] = PARSING_COLOR_LIST[label_idx]
    if fuse_img is not None:
        if isinstance(fuse_img, str):
            fuse_img = cv2.cvtColor(cv2.imread(fuse_img), cv2.COLOR_BGR2RGB)
            fuse_img = cv2.resize(fuse_img, img.shape[:2], interpolation=cv2.INTER_NEAREST)
        result = result * 0.5 + fuse_img * 0.5
        result = result.astype(np.uint8)
    return result


def naive_transfer(hair_parsing, target_parsing):
    """
    -1 means remove hair
    :param hair_parsing:
    :param target_parsing:
    :return:
    """
    target_parsing = target_parsing.copy().astype('int')
    target_parsing[target_parsing == HAIR_IDX] = 255
    target_parsing[hair_parsing == HAIR_IDX] = HAIR_IDX
    return target_parsing


def wrap_transform(landmarks_source, landmarks_target, source_img):
    return triangle_wrap_hair.wrap(landmarks_source, landmarks_target, source_img,
                                   draw_triangle=False)


def affine_transform(landmarks_source, landmarks_target, source_img):
    landmarks_source = np.concatenate([landmarks_source, np.ones((landmarks_source.shape[0], 1))], axis=1)
    affine_matrix = np.linalg.lstsq(landmarks_source, landmarks_target)[0].T
    return cv2.warpAffine(source_img, affine_matrix, dsize=source_img.shape[:2])


def hair_mask_transfer_wrap(hair_parsing, face_parsing, hair_lm_81, face_lm_81,
                            warp_temp_folder=None):
    """
    transfer according affine matrix with chosed landmarks
    :param face_lm_81: [0, 1]
    :param hair_lm_81: [0, 1]
    :param face_root_dir:
    :param hair_root_dir:
    :param need_transformed_hair: fast mode for data loader
    :param hair_parsing:
    :param face_parsing:
    :param hair_base_path:
    :param face_base_path:
    :return:
    """
    hair_mask = (hair_parsing == HAIR_IDX)
    parsing_shape = hair_parsing.shape[0]

    hair_lm_81 = hair_lm_81 * parsing_shape
    face_lm_81 = face_lm_81 * parsing_shape

    # chosen_landmarks = [0, 75, 68, 70, 80, 73, 74, 16]
    # chosen_landmarks = [0, 1, 2, 3, 4, 5, 16, 15, 14, 13, 12, 11, 77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]
    chosen_landmarks = list([kk for kk in range(81) if kk not in (26, 17, 25, 19)])

    filter_landmarks = chosen_landmarks

    hair_landmarks = hair_lm_81[filter_landmarks, :]
    target_landmarks = face_lm_81[filter_landmarks, :]
    hair_mask = hair_mask.astype('uint8')

    ### expand board
    bg_width = 80
    if bg_width > 0:
        hair_mask_total = np.zeros((512 + bg_width * 2, 512 + bg_width * 2), dtype='int')
        hair_mask_total[bg_width:512 + bg_width, bg_width:512 + bg_width] = hair_mask

        hair_mask_total[bg_width - 10: bg_width, hair_mask_total[bg_width, :] == 1] = 1
        hair_mask_total[-bg_width: -bg_width + 10, hair_mask_total[-1 - bg_width, :] == 1] = 1
        hair_mask_total[hair_mask_total[:, bg_width] == 1, bg_width - 10: bg_width] = 1
        hair_mask_total[hair_mask_total[:, -1 - bg_width] == 1, -bg_width: -bg_width + 10] = 1

        hair_mask = hair_mask_total
        hair_landmarks += bg_width
        target_landmarks += bg_width

    UV, triangle_path, wrap_name = triangle_wrap_hair.get_wrap_UV(hair_landmarks, target_landmarks, hair_mask,
                                                                  warp_temp_folder)
    transformed_hair_mask, hair_mask_triangle1, hair_mask_triangle2 = triangle_wrap_hair.wrap_by_uv(
        UV, triangle_path, wrap_name, hair_landmarks, target_landmarks, hair_mask)
    transformed_hair_mask = transformed_hair_mask.astype('uint8')

    if bg_width > 0:
        transformed_hair_mask = transformed_hair_mask[bg_width:-bg_width, bg_width:-bg_width]

    result_parsing = naive_transfer(transformed_hair_mask * HAIR_IDX, face_parsing)
    return result_parsing, {'hair_mask': transformed_hair_mask}


def wrap_for_image_with_idx(root_dir, hair_dir, face_dir, hair, face, wrap_temp_folder=None):
    """
    swap for ffhq or ffhq-liked imgs that ordered and preprocessed
    :param hair_root_dir:
    :param face_root_dir:
    :param hair:
    :param face:
    :param wrap_temp_folder:
    :param fully:
    :return:
    """
    hair_root_dir = os.path.join(root_dir, hair_dir)
    face_root_dir = os.path.join(root_dir, face_dir)
    hair_parsing = get_parsing_label(hair_root_dir, hair)
    face_parsing = get_parsing_label(face_root_dir, face)

    if LANDMARK_81_DIC is None:
        load_landmark_dic()
    hair_key = '%s___%s' % (hair_dir, hair[:-4])
    face_key = '%s___%s' % (face_dir, face[:-4])
    hair_lm_81 = LANDMARK_81_DIC[hair_key]
    face_lm_81 = LANDMARK_81_DIC[face_key]

    result_parsing, others_result_dict = hair_mask_transfer_wrap(hair_parsing, face_parsing,
                                                                 hair_lm_81, face_lm_81,
                                                                 warp_temp_folder=wrap_temp_folder)
    return result_parsing, others_result_dict


def wrap_by_imgs(hair_img_rgb, face_img_rgb, wrap_temp_folder=None, need_crop=True, verbose_print=False,
                 debug_mode=False, hair_parsing=None, face_parsing=None):
    if verbose_print:
        print('Generate face parsing for face image and hair image...')
    predictor_68 = predictor_dict[68]
    predictor_81 = predictor_dict[81]

    # TODO: exception for the situation of no face
    face_bbox = detector(face_img_rgb, 0)[0]
    face_lm_68 = np.array([[p.x, p.y] for p in predictor_68(face_img_rgb, face_bbox).parts()])

    if need_crop:
        from external_code.crop import recreate_aligned_images
        crop_size = 1024

        crop_face_img_pil, face_lm_68 = recreate_aligned_images(face_img_rgb, face_lm_68, output_size=crop_size)
        face_img_rgb = np.array(crop_face_img_pil)
        face_path = os.path.join(wrap_temp_folder, 'crop_%s' % os.path.basename('temp_face'))
        write_rgb(face_path, face_img_rgb)

        hair_bbox = detector(hair_img_rgb, 0)[0]
        hair_lm_68 = np.array([[p.x, p.y] for p in predictor_68(hair_img_rgb, hair_bbox).parts()])
        crop_hair_img_pil, hair_lm_68 = recreate_aligned_images(hair_img_rgb, hair_lm_68, output_size=crop_size)
        hair_img_rgb = np.array(crop_hair_img_pil)
        hair_path = os.path.join(wrap_temp_folder, 'crop_%s' % os.path.basename('temp_hair'))
        write_rgb(hair_path, hair_img_rgb)

    face_lm_81 = np.array([[p.x, p.y] for p in predictor_81(face_img_rgb, detector(face_img_rgb, 0)[0]).parts()])
    face_lm_81 = face_lm_81 / face_img_rgb.shape[1]
    hair_lm_81 = np.array([[p.x, p.y] for p in predictor_81(hair_img_rgb, detector(hair_img_rgb, 0)[0]).parts()])
    hair_lm_81 = hair_lm_81 / hair_img_rgb.shape[1]

    if hair_parsing is None:
        hair_parsing, hair_img = FaceParsing.parsing_img(hair_img_rgb)
        hair_parsing = FaceParsing.swap_parsing_label_to_celeba_mask(hair_parsing)
    if face_parsing is None:
        face_parsing, face_img = FaceParsing.parsing_img(face_img_rgb)
        face_parsing = FaceParsing.swap_parsing_label_to_celeba_mask(face_parsing)

    # TODO: 3ddfa no face exception
    if verbose_print:
        print("Adapt the target mask to the input face...")
    result_parsing, others_result_dict = hair_mask_transfer_wrap(hair_parsing, face_parsing,
                                                                 hair_lm_81, face_lm_81,
                                                                 warp_temp_folder=wrap_temp_folder)
    return result_parsing, others_result_dict
