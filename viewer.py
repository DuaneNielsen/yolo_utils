from matplotlib import pyplot as plt
import argparse
import yolo
from PIL import Image
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors


class Display:
    def __init__(self, ds, split):
        self.indx = 0
        self.fig, self.axes = plt.subplots(1, 1)
        self.img_ax = self.axes
        self.ds = ds
        self.split = split
        self.fig.subplots_adjust(bottom=0.2)
        self.img_ax.set_title(f'split: {split}')

        image_file, labels = self.ds.splits[split][self.indx]
        image = Image.open(image_file)

        self.im = self.img_ax.imshow(image)
        self.patches = []
        self.text = []
        self.draw_box(labels, image)
        self.axprev = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])

    def draw_box(self, labels, image):
        for patch in self.patches:
            patch.remove()
        self.patches = []
        for txt in self.text:
            txt.remove()
        self.text = []

        for l in labels.labels:
            tx, ty, w, h = l.tl_w_h(image.width, image.height)
            if l.label < len(mcolors.CSS4_COLORS):
                color = mcolors.CSS4_COLORS[list(mcolors.CSS4_COLORS)[l.label]]
            else:
                color = mcolors.CSS4_COLORS[list(mcolors.CSS4_COLORS)[0]]
            rect = Rectangle((tx, ty), w, h, linewidth=1, edgecolor=color, facecolor='none', label=f'{l.label} {l.name}')
            self.img_ax.add_patch(rect)
            self.patches.append(rect)
            txt = self.img_ax.text(tx, ty, f'{l.label} {l.name}', backgroundcolor=color)
            self.text.append(txt)

    def update(self):
        image_file, labels = self.ds.splits[self.split][self.indx]
        image = Image.open(image_file)
        self.img_ax.set_title(labels.filename)
        self.im.set_data(image)
        self.im.set_extent((0, image.width, image.height, 0))

        self.draw_box(labels, image)
        self.fig.canvas.draw_idle()

    def next(self, args):
        if self.indx < len(ds.splits[self.split])-1:
            self.indx += 1
        self.update()

    def prev(self, args):
        if self.indx >= 1:
            self.indx -= 1
        self.update()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('split', choices=['train', 'val', 'test'], default='train')
    args = parser.parse_args()

    data_yaml = args.directory + '/data.yaml'

    ds = yolo.YoloDataset(data_yaml)

    if len(ds[args.split]) > 0:
        display = Display(ds, args.split)
        bnext = Button(display.axnext, 'Next')
        bnext.on_clicked(display.next)
        bprev = Button(display.axprev, 'Previous')
        bprev.on_clicked(display.prev)
        plt.show()
    else:
        print(f"datset does not contain {args.split} split")