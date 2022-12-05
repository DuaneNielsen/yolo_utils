# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import shutil

from yolo import YoloDataset
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_yaml')
    # parser.add_argument('classes')
    # parser.add_argument('dest')
    args = parser.parse_args()

    ds = YoloDataset(args.data_yaml)

    items_copied = 0
    if ds.train:
        pbar = tqdm(ds.train)
        for image, labels in pbar:
            pbar.set_description(f'copying train set, copied: {items_copied}')
            labels_f = labels.filter({0, 1})
            if len(labels_f) > 0:
                labels_f.write('/mnt/data/data/cones_subset/train/labels')
                shutil.copy(image, '/mnt/data/data/cones_subset/train/images')
                items_copied += 1

    # if ds.val:
    #     for image, labels in tqdm(ds.val):
    #         labels_f = labels.filter({0, 1})
    #         if len(labels_f) > 0:
    #             labels_f.write('/mnt/data/data/cones_subset/val/labels')
    #             shutil.copy(image, '/mnt/data/data/cones_subset/val/images')
