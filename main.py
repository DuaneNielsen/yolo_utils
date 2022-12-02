# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import yaml
from pathlib import Path
from copy import deepcopy
import warnings
from tqdm import tqdm


class YoloLabels:
    def __init__(self, line):
        self.label = int(line.split()[0])
        _, self.x, self.y, self.w, self.h = map(float, line.split())

    def __str__(self):
        return f'{self.label} {self.x} {self.y} {self.w} {self.h}'


class YoloLabelFile:
    def __init__(self, filename):
        self.filename = filename
        self.labels = []
        self.label_count = {}
        with open(filename) as f:
            for line in f:
                lbl = YoloLabels(line)
                if lbl.label in self.label_count:
                    self.label_count[lbl.label] += 1
                else:
                    self.label_count[lbl.label] = 1
                self.labels.append(lbl)

    def __str__(self):
        return ''.join([str(label) + '\n' for label in self.labels])

    def __repr__(self):
        return self.filename

    def __len__(self):
        return len(self.labels)

    def filter(self, labels_to_filter_out):
        """
        :param labels_to_filter_out: set of labels to filter out
        :return: a YoloLabelFile
        """
        minime = deepcopy(self)
        i = 0
        while i < len(minime.labels):
            if minime.labels[i].label in labels_to_filter_out:
                del minime.labels[i]
            i += 1
        return minime

    def write(self, dir):
        """
        write the image labels to a file
        :param dir: the directory to write to
        """
        filename = Path(dir) / Path(self.filename).name
        with filename.open('w') as f:
            f.write(str(self))


class YoloSet:
    def __init__(self, name, image_folder):
        self.name = name
        self.image_folder = image_folder
        self.images_without_label = 0
        self.objects_by_class = {}
        self.data = self.build()

    def build(self):
        files = Path(self.image_folder).glob("*")

        def label_from_image(image_path):
            return Path(image_path).parent.parent / 'labels' / (image_path.stem + '.txt')

        set = []
        for image in files:
            label = label_from_image(image)
            if label.exists():
                img, lblf = str(image), YoloLabelFile(str(label))
                for lb, count in lblf.label_count.items():
                    if lb in self.objects_by_class:
                        self.objects_by_class[lb] += count
                    else:
                        self.objects_by_class[lb] = count
                set.append((img, lblf))
            else:
                self.images_without_label += 1
                warnings.warn("Warning: image in the dataset but no file, this is OK if it happens infrequently")

        return set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return str(self.__dict__)

    def print_stats(self):
        print(f'length : {len(ds.train)}')
        print(f'images without label: {self.images_without_label}')
        for lbl, count in self.objects_by_class.items():
            print(f'{lbl}: {count}')


class YoloDataset:
    def __init__(self, data_yaml_path):
        self.dir = Path(args.data_yaml).parent

        def filter_silly_paths(silly_path):
            return silly_path[3:]

        # change to root directory of dataset
        with open(data_yaml_path) as data_yaml:
            try:
                data_yaml = yaml.safe_load(data_yaml)
                self.train = YoloSet('train', str(self.dir / filter_silly_paths(
                    data_yaml['train']))) if 'train' in data_yaml else None
                self.test = YoloSet('test', str(self.dir / filter_silly_paths(
                    data_yaml['test']))) if 'test' in data_yaml else None
                self.val = YoloSet('val',
                                   str(self.dir / filter_silly_paths(data_yaml['val']))) if 'val' in data_yaml else None
                self.nc = data_yaml['nc']
                self.names = data_yaml['names'] if 'names' in data_yaml else None
            except yaml.YAMLError as exc:
                print(exc)
                raise Exception('could not load yaml file')

    def __repr__(self):
        return str(self.__dict__)


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