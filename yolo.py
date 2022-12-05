import warnings
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import yaml


class YoloLabels:
    def __init__(self, line, dataset):
        self.label = int(line.split()[0])
        _, self.cx, self.cy, self.w, self.h = map(float, line.split())
        self.dataset = dataset
        if self.label < len(self.dataset.names):
            self.name = self.dataset.names[self.label]
        else:
            self.name = f'{self.label}'

    def __str__(self):
        return f'{self.label} {self.cx} {self.cy} {self.w} {self.h}'

    def tl_w_h(self, dw, dh):
        """Top left, width height """
        tx = int((self.cx - self.w / 2) * dw)
        ty = int((self.cy - self.h / 2) * dh)
        return tx, ty, self.w * dw, self.h * dh

class YoloLabelFile:
    def __init__(self, filename, dataset):
        self.filename = filename
        self.labels = []
        self.label_count = {}
        self.dataset = dataset
        with open(filename) as f:
            for line in f:
                lbl = YoloLabels(line, dataset)
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
    def __init__(self, name, image_folder, dataset):
        self.name = name
        self.image_folder = image_folder
        self.images_without_label = 0
        self.objects_by_class = {}
        self.dataset = dataset
        self.data = self.build()

    def build(self):
        files = Path(self.image_folder).glob("*")

        def label_from_image(image_path):
            return Path(image_path).parent.parent / 'labels' / (image_path.stem + '.txt')

        set = []
        pbar = tqdm(list(files))
        for image in pbar:
            pbar.set_description(f'reading {self.name} set ...')
            label = label_from_image(image)
            if label.exists():
                img, lblf = str(image), YoloLabelFile(str(label), self.dataset)
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
        print(f'length : {len(self)}')
        print(f'images without label: {self.images_without_label}')
        for lbl, count in self.objects_by_class.items():
            print(f'{lbl}: {count}')


class YoloDataset:
    def __init__(self, data_yaml_path):
        self.dir = Path(data_yaml_path).parent
        self.sets = {}

        def filter_silly_paths(silly_path):
            return silly_path[3:]

        # change to root directory of dataset
        with open(data_yaml_path) as data_yaml:
            try:
                data_yaml = yaml.safe_load(data_yaml)
                self.names = data_yaml['names'] if 'names' in data_yaml else None
                self.sets['train'] = YoloSet('train', str(self.dir /
                    data_yaml['train']), self) if 'train' in data_yaml else None
                self.sets['test'] = YoloSet('test', str(self.dir /
                    data_yaml['test']), self) if 'test' in data_yaml else None
                self.sets['val'] = YoloSet('val',
                                   str(self.dir / data_yaml['val']), self) if 'val' in data_yaml else None
                self.nc = data_yaml['nc']

            except yaml.YAMLError as exc:
                print(exc)
                raise Exception('could not load yaml file')

    @property
    def train(self):
        return self.sets['train']

    @property
    def test(self):
        return self.sets['test']

    @property
    def val(self):
        return self.sets['val']


    def __repr__(self):
        return str(self.__dict__)
