import os
import util.rng
from util.data_loader import load_image, load_object
from util.pascal_voc_io import PascalVocWriter
import cv2
from copy import deepcopy


class Generator(object):
    def __init__(self, args):
        self.args = args
        self.objects = {}
        self.target_size = (args.width, args.height)
        self.out_image_path = None
        self.out_anno_path = None
        self.background = None

        # set seed for random generator
        util.rng.init(args.rand_seed + args.start_idx)


    # Main procedure
    # input :
    # single_image_process : must return image and annotation.
    def run(self, single_image_process):
        self.load()

        self.make_outdir()

        end_idx = self.args.start_idx + self.args.outputs

        print('start making data.')
        for i in range(self.args.start_idx, end_idx):
            image, annotation = single_image_process()

            self.write_output('{0:05d}'.format(i), image, annotation)

            # print log
            if (i + 1) % 10 == 0:
                print('{} of {}'.format(i + 1, end_idx))

        print('done.')


    # Load source data
    def load(self):
        assert self.args.data_dir is not None

        print('Load source data.')
        self.objects = load_object(self.args.data_dir)

        if self.args.bg_dir is not None:
            print('Load background.')
            self.background = load_image(self.args.bg_dir)

    # Created dir to store outputs
    def make_outdir(self):
        def make_dir(path):
            if not os.path.exists(path):
                os.mkdir(path)

        out_dir = os.path.expanduser(self.args.out_dir)
        self.out_image_path = os.path.join(out_dir, 'images')
        self.out_anno_path = os.path.join(out_dir, 'annotations')

        make_dir(out_dir)
        make_dir(self.out_image_path)
        make_dir(self.out_anno_path)

    def write_output(self, file_title, image, annotation):
        assert image is not None, annotation is not None

        # Write image
        cv2.imwrite(os.path.join(self.out_image_path, file_title + '.jpg'), image)

        folder_name = os.path.join(self.args.xml_folder, 'images')

        voc_writer = PascalVocWriter(
            folder_name,
            file_title + '.jpg',
            (self.args.height, self.args.width, 3),
            databaseSrc=self.args.db_title
        )

        for bbox in annotation:
            voc_writer.addBndBox2(bbox)

        voc_writer.save(os.path.join(self.out_anno_path, file_title + '.xml'))