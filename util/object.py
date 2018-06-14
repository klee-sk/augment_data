import cv2
from math import sqrt
from copy import deepcopy
import util.rng
import util.image

def resize_area(_object, area_lower, area_upper):
    rng = util.rng.get()

    shape = _object['image'].shape
    object_area = shape[0] * shape[1]

    # if area in (area_lower, area_upper), reshape with prob of 0.5.
    if object_area >= area_lower and object_area <= area_upper:
        if rng.random() < 0.5:
            return

    area_ratio = rng.uniform(area_lower, area_upper) / float(object_area)
    ratio = sqrt(area_ratio)

    _object['image'] = cv2.resize(_object['image'], (int(round(shape[1] * ratio)), int(round(shape[0] * ratio))))

    # resize points of bbox
    for key in ('xmin', 'ymin', 'xmax', 'ymax'):
        _object['bbox'][key] = int(round(_object['bbox'][key] * ratio))


def pick(objects, count_range=(4, 8)):
    rng = util.rng.get()
    count = rng.randint(*count_range)

    return [deepcopy(rng.choice(rng.choice(objects))) for _ in range(0, count)]


def transform_shape(_object):
    TRANSFORM_PROB = 0.5

    rng = util.rng.get()

    image = _object['image']

    # horizontal flip
    if rng.random() < TRANSFORM_PROB:
        image = cv2.flip(image, 0)

    # vertical flip
    if rng.random() < TRANSFORM_PROB:
        image = cv2.flip(image, 1)

    # rotation
    if rng.random() < TRANSFORM_PROB:
        bbox = _object['bbox']
        height, width, _ = image.shape

        if rng.random() < 0.5: # Clock-wise
            rotate_code = cv2.ROTATE_90_CLOCKWISE

            xmax = bbox['xmax']
            bbox['xmax'] = height - bbox['ymin']
            bbox['ymin'] = bbox['xmin']
            bbox['xmin'] = height - bbox['ymax']
            bbox['ymax'] = xmax

        else: # CCW
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

            ymin = bbox['ymin']
            bbox['ymin'] = width - bbox['xmax']
            bbox['xmax'] = bbox['ymax']
            bbox['ymax'] = width - bbox['xmin']
            bbox['xmin'] = ymin

        _object['image'] = cv2.rotate(image, rotateCode=rotate_code)

def bbox_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bbox1 : dict
        Keys: {'x1', 'xmax', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (xmax, y2) position is at the bottom right corner
    bbox2 : dict
        Keys: {'x1', 'xmax', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (xmax, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bbox1['xmin'] < bbox1['xmax']
    assert bbox1['ymin'] < bbox1['ymax']
    assert bbox2['xmin'] < bbox2['xmax']
    assert bbox2['ymin'] < bbox2['ymax']

    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1['xmin'], bbox2['xmin'])
    y_top = max(bbox1['ymin'], bbox2['ymin'])
    x_right = min(bbox1['xmax'], bbox2['xmax'])
    y_bottom = min(bbox1['ymax'], bbox2['ymax'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bbox1['xmax'] - bbox1['xmin']) * (bbox1['ymax'] - bbox1['ymin'])
    bb2_area = (bbox2['xmax'] - bbox2['xmin']) * (bbox2['ymax'] - bbox2['ymin'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0 or iou <= 1.0
    return iou


def dispose_objects(objects, background):

    def try_desposing(object, background_shape, annotation, trial_count=20):
        def is_overlap(box, boxes):
            for b in boxes:
                if bbox_iou(box, b) > 0:
                    return True
            return False

        bg_height, bg_width, _ = background_shape
        image_height, image_width, _ = object['image'].shape

        x_margin = bg_width - image_width
        y_margin = bg_height - image_height

        if x_margin > 0 and y_margin > 0:
            for _ in range(0, trial_count):
                offset_x = rng.randint(0, bg_width - image_width)
                offset_y = rng.randint(0, bg_height - image_height)

                moved_image_box = {
                    'xmin': offset_x,
                     'xmax': offset_x + width,
                     'ymin': offset_y,
                     'ymax': offset_y + height,
                }

                if is_overlap(moved_image_box, annotation) == False:
                    # brack loop and return offset
                    return offset_x, offset_y

        elif x_margin == 0 and y_margin == 0:
            # image and backgroud have same shape
            return 0, 0

        # image is bigger than background or failed to get offset for disposed image
        return None, None

    rng = util.rng.get()

    out_image = deepcopy(background)

    annotation = []

    for object in objects:
        bbox = object['bbox']
        height, width, _ = object['image'].shape

        offset_x, offset_y = try_desposing(object, background.shape, annotation)

        if offset_x is not None and offset_y is not None:
            # 1. move bbox with offset and add to annotation
            bbox['xmin'] += offset_x
            bbox['xmax'] += offset_x
            bbox['ymin'] += offset_y
            bbox['ymax'] += offset_y

            annotation.append(bbox)

            # 2. alpha blending
            util.image.alpha_blending(object['image'], out_image[offset_y: offset_y + height, offset_x: offset_x + width], (3, 3))


    return out_image, annotation
