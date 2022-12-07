# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import shutil
from pathlib import Path

import yolo
from yolo import YoloDataset
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='copy a subset yolo dataset to target directory'
                    'you can copy a subset and also remap classes')
    parser.add_argument('src_dir', help='directory of the source dataset')
    parser.add_argument('dest', help='the directory of the dataset to copy to')
    parser.add_argument('label_map', type=str, nargs='+',
                        help='list of src_label:dest_label to copy'
                        'ie; 0:1 will copy only class 0 from source dataset to class 1')
    args = parser.parse_args()

    data_yaml = args.src_dir + '/data.yaml'

    whitelist, label_map = [], {}
    for mapping in args.label_map:
        tokens = mapping.split(':')
        if len(tokens) == 2:
            src_lbl, dst_lbl = map(int, tokens)
        else:
            src_lbl, dst_lbl = int(tokens[0]), int(tokens[0])
        whitelist += [src_lbl]
        label_map[src_lbl] = dst_lbl

    ds = YoloDataset(data_yaml)
    yolo.create_skeleton(args.dest)

    def copy_set(ds, dest, name, label_dict, label_map):
        items_copied = 0
        if ds.splits[name]:
            pbar = tqdm(ds[name])
            for image, labels in pbar:
                pbar.set_description(f'copying {name} set, copied: {items_copied}')
                labels_f = labels.filter(whitelist=label_dict).map(label_map)
                if len(labels_f) > 0:
                    labels_f.write(f'{dest}/{name}/labels')
                    shutil.copy(image, f'{dest}/{name}/images')
                    items_copied += 1

    copy_set(ds, args.dest, 'train', whitelist, label_map)
    copy_set(ds, args.dest, 'test', whitelist, label_map)
    copy_set(ds, args.dest, 'val', whitelist, label_map)
