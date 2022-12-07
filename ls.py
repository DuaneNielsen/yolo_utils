# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse


from rich.console import Console
from rich.table import Table
from yolo import YoloDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()

    data_yaml = args.directory + '/data.yaml'

    ds = YoloDataset(data_yaml)

    for split in ds.splits:
        table = Table(title=f"{split} split")

        table.add_column("Label", style="cyan", no_wrap=True)
        table.add_column("Name", style="magenta")
        table.add_column("Object Count", justify="right", style="green")
        table.add_column("Images", justify="right", style="green")

        lbls = sorted([lbl for lbl in ds[split].objects_by_class])
        for lbl in lbls:
            object_count = ds[split].objects_by_class[lbl]
            image_count = ds[split].images_with_object[lbl]
            name = ds.lookup_name(lbl)
            table.add_row(f"{lbl}", name, f'{object_count}', f'{image_count}')

        console = Console()
        console.print(table)