import warnings
from pathlib import Path
from tqdm import tqdm
import yaml
from copy import copy


def create_skeleton(fn):
    # create the directories
    Path(fn).mkdir(parents=True, exist_ok=True)
    (Path(fn) / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (Path(fn) / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (Path(fn) / 'test' / 'labels').mkdir(parents=True, exist_ok=True)
    (Path(fn) / 'test' / 'images').mkdir(parents=True, exist_ok=True)
    (Path(fn) / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
    (Path(fn) / 'val' / 'images').mkdir(parents=True, exist_ok=True)

    # create the data.yaml
    if not (Path(fn)/'data_yaml').exists():

        data_yaml = {
            'train': './train/images',
            'test': './test/images',
            'val': './val/images',
        }

        with open(f'{fn}/data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)


class YoloLabels:
    def __init__(self, line, names):
        self.label = int(line.split()[0])
        self.names = names
        _, self.cx, self.cy, self.w, self.h = map(float, line.split())
        if names is not None:
            if self.label < len(self.names):
                self.name = self.names[self.label]
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
    def __init__(self, filename, names):
        self.filename = filename
        self.labels = []
        self.label_count = {}
        self.names = names
        with open(filename) as f:
            for line in f:
                lbl = YoloLabels(line, names)
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

    def filter(self, blacklist=None, whitelist=None):
        """
        :param labels_to_filter_out: set of labels to filter out
        :return: a YoloLabelFile
        """
        assert not (blacklist is not None and whitelist is not None), 'only supply a blacklist OR a whitelist'

        minime = copy(self)
        i = 0
        while i < len(minime.labels):
            if blacklist is not None:
                if minime.labels[i].label in blacklist:
                    del minime.labels[i]
                else:
                    i += 1
            if whitelist is not None:
                if minime.labels[i].label not in whitelist:
                    del minime.labels[i]
                else:
                    i += 1
        return minime

    def map(self, map):
        minime = copy(self)
        for label in minime.labels:
            if label in map:
                label.label = map[label]
        return minime

    def write(self, dir):
        """
        write the image labels to a file
        :param dir: the directory to write to
        """
        filename = Path(dir) / Path(self.filename).name
        with filename.open('w') as f:
            f.write(str(self))


class YoloSplit:
    def __init__(self, split, names):
        self.name = split
        self.image_folder = None
        self.images_without_label = 0
        self.objects_by_class = {}
        self.images_with_object = {}
        self.names = names

    def load(self, image_folder):
        self.image_folder = image_folder
        files = Path(self.image_folder).glob("*")

        def label_from_image(image_path):
            return Path(image_path).parent.parent / 'labels' / (image_path.stem + '.txt')

        set = []
        pbar = tqdm(list(files))
        for image in pbar:
            pbar.set_description(f'reading {self.name} set ...')
            label = label_from_image(image)
            if label.exists():
                img, lblf = str(image), YoloLabelFile(str(label), self.names)
                for lb, count in lblf.label_count.items():
                    if lb in self.objects_by_class:
                        self.objects_by_class[lb] += count
                        self.images_with_object[lb] += 1
                    else:
                        self.objects_by_class[lb] = count
                        self.images_with_object[lb] = 1

                set.append((img, lblf))
            else:
                self.images_without_label += 1
                warnings.warn("Warning: image in the dataset but no file, this is OK if it happens infrequently")
            self.data = set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return str(self.__dict__)


class YoloDataset:
    def __init__(self, data_yaml_path):
        self.dir = Path(data_yaml_path).parent
        self.splits = {}

        def filter_silly_paths(silly_path):
            if silly_path[0:2] == '..':
                return silly_path[3:]
            else:
                return silly_path

        # change to root directory of dataset
        with open(data_yaml_path) as data_yaml:
            try:
                data_yaml = yaml.safe_load(data_yaml)
                self.names = data_yaml['names'] if 'names' in data_yaml else None

                def load_split(name):
                    split = YoloSplit(name, self.names)
                    if name in data_yaml:
                        path = str(self.dir / filter_silly_paths(data_yaml[name]))
                        split.load(path)
                    self.splits[name] = split

                load_split('train')
                load_split('test')
                load_split('val')
                self.nc = data_yaml['nc'] if 'nc' in data_yaml else None

            except yaml.YAMLError as exc:
                print(exc)
                raise Exception('could not load yaml file')

    def __getitem__(self, item):
        return self.splits[item]

    @property
    def train(self):
        return self.splits['train']

    @property
    def test(self):
        return self.splits['test']

    @property
    def val(self):
        return self.splits['val']

    def lookup_name(self, label):
        if self.names is not None:
            if label < len(self.names):
                return self.names[label]
        else:
            return ''

    def __repr__(self):
        return str(self.__dict__)
