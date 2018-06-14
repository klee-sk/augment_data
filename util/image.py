import numpy as np
import cv2
import util.rng

def roi_crop(image, roi):
    ymin = roi['ymin']
    ymax = roi['ymax']
    xmin = roi['xmin']
    xmax = roi['xmax']

    assert xmin < xmax and ymin < ymax, roi

    return image[ymin:ymax, xmin:xmax]

def random_crop(images, target_size):
    rng = util.rng.get()

    # 1. Choose randomly
    source_image = rng.choice(images)
    image_size = (source_image.shape[1], source_image.shape[0])

    # 2. Crop with resizing
    size = []
    offset = []

    # min( w_source / w_target, h_source / h_target)
    min_ratio = min([float(image_size[i]) / float(target_size[i]) for i in range(0, 2)])

    if min_ratio <= 0:
        resize_ratio = min_ratio
    else:
        resize_ratio = rng.uniform(1.0, min_ratio)

    for i in range(0, 2):
        length = int(round(resize_ratio * target_size[i]))
        length = min(length, image_size[i])

        size.append(length)
        margin = image_size[i] - length

        if margin is 0:
            gap = 0
        else:
            gap = rng.randrange(0, margin)
        offset.append(gap)

    roi = {
        'xmin': offset[0],
        'ymin': offset[1],
        'xmax': offset[0] + size[0],
        'ymax': offset[1] + size[1],
    }

    return cv2.resize(roi_crop(source_image, roi), target_size, cv2.INTER_CUBIC)

def convert_bgr_to_hsv(bgr):
    if bgr.shape[2] == 3:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # else is shape[2] == 4
    hsv = np.ndarray(shape=bgr.shape, dtype=bgr.dtype)
    hsv[:, :, 3] = bgr[:, :, 3]
    hsv[:, :, :3] = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2HSV)

    return hsv

def convert_hsv_to_bgr(hsv):
    if hsv.shape[2] == 3:
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # else is shape[2] == 4
    bgr = np.ndarray(shape=hsv.shape, dtype=hsv.dtype)
    bgr[:, :, 3] = hsv[:, :, 3]
    bgr[:, :, :3] = cv2.cvtColor(hsv[:, :, :3], cv2.COLOR_HSV2BGR)

    return bgr

def convert_scale_abs(image, scale_lower=1, scale_upper=1, abs_delta = 0):
    rng = util.rng.get()

    alpha = 1
    beta = 0

    if scale_lower != 1 or scale_upper != 1:
        alpha = rng.uniform(scale_lower, scale_upper)

    if abs_delta != 0:
        beta = rng.randint(0, abs_delta)

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def distort_hsv(bgr_image, param):
    rng = util.rng.get()

    hsv = convert_bgr_to_hsv(bgr_image)

    if rng.random() < param['hue_prob']:
        hsv[:, :, 0] = convert_scale_abs(hsv[:, :, 0], abs_delta=param['hue_delta'])

    if rng.random() < param['saturation_prob']:
        hsv[:, :, 2] = convert_scale_abs(
            hsv[:, :, 2],
            scale_lower=param['saturation_lower'],
            scale_upper=param['saturation_upper']
        )

    if rng.random() < param['value_prob']:
        hsv[:, :, 2] = convert_scale_abs(
            hsv[:, :, 2],
            scale_lower=param['value_lower'],
            scale_upper=param['value_upper']
        )

    return convert_hsv_to_bgr(hsv)

def alpha_blending(source_image, target_image, blur_size=(3, 3)):
    background = target_image[:, :, :3].astype(float)
    foreground = source_image[:, :, :3].astype(float)
    alpha = cv2.blur(source_image[:, :, 3], blur_size)

    alpha = np.tile(alpha[:, :, None], [1, 1, 3])

    alpha = alpha.astype(float) / 255.0

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    merge = cv2.add(foreground, background)
    target_image[:, :, :3] = merge.astype(np.uint8)



