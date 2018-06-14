from util.arguments import parser
from util.generator import Generator
import util.object

import util.image
import cv2

class ExcavatorData(Generator):
    def __init__(self, args):
        self.distort_param = {
                'hue_prob': 0.5,
                'hue_delta': 18,
                'saturation_prob': 0.5,
                'saturation_lower': 0.5,
                'saturation_upper': 1.5,
                'value_prob': 0.5,
                'value_lower': 0.5,
                'value_upper': 1.5,
                }
        super().__init__(args)

    def process(self):
        # objects data : self.objects = {'image', 'bbox'}
        # background data : self.background
        # self.target_size = (args.width, args.height)

        # Must return image and annotation

        # 1. crop background
        bg = util.image.random_crop(self.background, self.target_size)
        # 2. distort background
        bg = util.image.distort_hsv(bg, self.distort_param)
        # 3. select objects randomly
        objects = util.object.pick(self.objects, (6, 12))

        for _object in objects:
            # 4. resize object w.r.t area
            util.object.resize_area(_object, 50*50, 200*200)

            # 5. transform object
            util.object.transform_shape(_object)
            _object['image'] = util.image.distort_hsv(_object['image'], self.distort_param)

        image, annotation = util.object.dispose_objects(objects, bg)

        return image, annotation






def main(args):
    data_generator = ExcavatorData(args)

    data_generator.run(data_generator.process)


if __name__ == '__main__':
    # argments are defined in 'util.arguments.py'
    parser.set_defaults(
        db_title='Excavator Dataset June18',
        data_dir='~/Data/excavator/objects',
        bg_dir='~/Data/excavator/bg',
        out_dir='~/Data/excavator/out',
        width=1920,
        height=1080,
    )
    main(parser.parse_args())


