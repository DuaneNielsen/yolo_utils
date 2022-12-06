# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import shutil
from pathlib import Path
from yolo import YoloDataset, create_skeleton
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_yaml')
    parser.add_argument('dest')
    parser.add_argument('whitelist', type=int, nargs='+')
    args = parser.parse_args()
    whitelist = {i for i in args.whitelist}

    ds = YoloDataset(args.data_yaml)

    Path(args.dest).mkdir(parents=True, exist_ok=True)
    (Path(args.dest)/'train'/'labels').mkdir(parents=True, exist_ok=True)
    (Path(args.dest)/'train'/'images').mkdir(parents=True, exist_ok=True)
    (Path(args.dest)/'test'/'labels').mkdir(parents=True, exist_ok=True)
    (Path(args.dest)/'test'/'images').mkdir(parents=True, exist_ok=True)
    (Path(args.dest)/'val'/'labels').mkdir(parents=True, exist_ok=True)
    (Path(args.dest)/'val'/'images').mkdir(parents=True, exist_ok=True)

    def copy_set(ds, dest, name, label_dict):
        items_copied = 0
        if ds.splits[name]:
            pbar = tqdm(ds.train)
            for image, labels in pbar:
                pbar.set_description(f'copying {name} set, copied: {items_copied}')
                labels_f = labels.filter(whitelist=label_dict)
                if len(labels_f) > 0:
                    labels_f.write(f'{dest}/{name}/labels')
                    shutil.copy(image, f'{dest}/{name}/images')
                    items_copied += 1

    copy_set(ds, args.dest, 'train', whitelist)
    copy_set(ds, args.dest, 'test', whitelist)
    copy_set(ds, args.dest, 'val', whitelist)
