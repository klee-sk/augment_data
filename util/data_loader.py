import os
import numpy as np
import cv2

# find image only
def load_image(_path):
    _path = os.path.expanduser(_path)
    assert os.path.exists(_path), '"{}" is not exist'.format(_path)

    images = []

    # search all files under _path
    file_names = os.listdir(_path)
    for file in sorted(file_names):
        file_path = os.path.join(_path, file)
        image = cv2.imread(file_path)

        if image is not None:
            images.append(image)
            print(file_path)

    assert len(images) != 0, 'No image is loaded'

    print('Done.')
    return images

def load_object(_path):
    _path = os.path.expanduser(_path)
    assert os.path.exists(_path), '"{}" is not exist'.format(_path)

    objects = []

    # find dir which has image of one kind of classes.
    dir_names = os.listdir(_path)
    for class_name in sorted(dir_names):
        dir_path = os.path.join(_path, class_name)

        if not os.path.isdir(dir_path):
            continue

        # class data
        class_data = []

        file_names = os.listdir(dir_path)

        for file in sorted(file_names):
            file_path = os.path.join(dir_path, file)

            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            if image is None:
                continue

            [h, w] = image.shape[:2]

            box = [[0, 0], [w, h]]
            bbox = {'name': class_name,
                    'xmin': box[0][0],
                    'ymin': box[0][1],
                    'xmax': box[1][0],
                    'ymax': box[1][1], }

            # if image channel == 3, make alpha channel filled with 255.
            if image.shape[2] == 3:
                image = np.concatenate((image, np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * 255),
                                       axis=2)

            class_data.append({'image': image, 'bbox': bbox})

            print(file_path)

        objects.append(class_data)
        # end of class data

    print('Done.')
    return objects


def load(data_path):
    data_path = os.path.expanduser(data_path)
    bg_dir = os.path.join(data_path, 'bg')
    objects_dir = os.path.join(data_path, 'objects')

    assert os.path.exists(bg_dir), '"{}" is not exist'.format(bg_dir)
    assert os.path.exists(objects_dir), '"{}" is not exist'.format(objects_dir)

    bg = load_image(bg_dir)
    objects = load_object(objects_dir)

    return {'background': bg,
            'objects': objects,}



