# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse

from yolo import YoloDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_yaml')
    args = parser.parse_args()

    ds = YoloDataset(args.data_yaml)

    print('Stats')
    print(ds.nc)
    print(ds.names)

    ds.train.print_stats()
    ds.val.print_stats()
    # if ds.train:
    #     for image, labels in tqdm(ds.train):
    #         labels.filter({0, 1}).write('/mnt/data/data/cones_aug_dataset/train/labels')
    # if ds.val:
    #     for image, labels in tqdm(ds.val):
    #         labels.filter({0, 1}).write('/mnt/data/data/cones_aug_dataset/val/labels')