import argparse
import yolo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    yolo.create_skeleton(args.directory)