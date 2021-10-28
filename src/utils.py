from pathlib import Path
import argparse


def valid_path(path):
    path = Path(path)
    if path.is_dir():
        return 1, path
    elif path.is_file():
        return 0, path
    else:
        raise argparse.ArgumentTypeError(
            f"{path} is not a valid path")
