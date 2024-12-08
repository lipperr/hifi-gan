import argparse
from src.utils.load_utils import download_best_model


def main(path):
    path = download_best_model(path)
    return path


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--path", default=None, type=str)
    args.parse_args()
    main(args)