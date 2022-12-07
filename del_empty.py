import argparse
from yolo import YoloDataset
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--dry-run', action='store_true', default=False)
    args = parser.parse_args()

    data_yaml = args.directory + '/data.yaml'

    ds = YoloDataset(data_yaml)

    for split in ds.splits:
        for img_f, lblf in ds.splits[split]:
            if len(lblf) == 0:
                if not args.dry_run:
                    os.remove(lblf.filename)
                    os.remove(img_f)
                else:
                    print(f'del {lblf.filename}')
                    print(f'del {img_f}')
