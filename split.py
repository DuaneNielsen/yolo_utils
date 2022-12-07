# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import random
import yolo
import shutil
from tqdm import tqdm


def copy_to_split(directory, indices, split):
    for item in indices:
        img_f, label = ds.train[item]
        label.write(directory + f'/{split}/labels')
        shutil.copy(img_f, directory + f'/{split}/images')


def move_to_split(directory, indices, split):
    indices = tqdm(indices)
    indices.set_description(f'moving images and labels to {split}')

    for item in indices:
        img_f, label = ds.train[item]
        shutil.move(label.filename, directory + f'/{split}/labels')
        shutil.move(img_f, directory + f'/{split}/images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--dry-run', action='store_true', default=False)
    args = parser.parse_args()

    data_yaml = args.directory + '/data.yaml'

    ds = yolo.YoloDataset(data_yaml)

    if len(ds.train) > 0 and len(ds.test) == 0 and len(ds.val) == 0:

        val_test = random.sample(list(range(len(ds.train))), len(ds.train)//5)
        test = val_test[:len(val_test)//2]
        val = val_test[len(val_test)//2:]

        print(f'no val or test split.. moving {len(test)} images to '
              f'test and {len(val)} images to val of {len(ds.train)} images')

        if not args.dry_run:
            yolo.create_skeleton(args.directory)
            move_to_split(args.directory, test, ds.get_path('test'))
            move_to_split(args.directory, val, ds.get_path('val'))

    elif len(ds.train) > 0 and len(ds.test) == 0 and len(ds.val) > 0:
        test = random.sample(list(range(len(ds.train))), len(ds.val))

        print(f'no test split.. moving {len(test)}/{len(ds.train)} images to test ')
        if not args.dry_run:
            yolo.create_skeleton(args.directory)
            move_to_split(args.directory, test, ds.get_path('test'))

    elif len(ds.train) > 0 and len(ds.test) > 0 and len(ds.val) == 0:
        val = random.sample(list(range(len(ds.train))), len(ds.test))
        print(f'no val split.. moving {len(val)}/{len(ds.train)} images to val ')
        yolo.create_skeleton(args.directory)
        move_to_split(args.directory, val, ds.get_path('val'))

    else:
        print('dataset contains all splits')

