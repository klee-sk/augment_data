import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--db_title',
    type=str,
)

parser.add_argument(
    '--xml_folder',
    type=str,
    default='~/dl_data'
)

parser.add_argument(
    '--data_dir',
    type=str,
)

parser.add_argument(
    '--bg_dir',
    type=str,
)

parser.add_argument(
    '--out_dir',
    type=str,
    default='data'
)


parser.add_argument(
    '--outputs',
    type=int,
    default=30000
)

parser.add_argument(
    '--width',
    type=int,
    default=640
)

parser.add_argument(
    '--height',
    type=int,
    default=480
)

parser.add_argument(
    '--rand_seed',
    type=int,
    default=11512
)

parser.add_argument(
    '--start_idx',
    type=int,
    default=0
)

parser.add_argument(
    '--resize_ratio',
    type=float,
    default=1.10
)

