import argparse
import dna2vec
import transformer
import validation
from pathlib import Path
from myLog import Log
from datetime import datetime


class DNAcom():
    """
    Main class: take the input and run the pipeline
    """

    def __init__(self):
        self.pipeline()

    def pipeline(self):
        pass


def arg_parse():
    """
    Parsing arguments function
    """
    parser = argparse.ArgumentParser(
        description='DNA compression')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-if', '--input_file',
                       type=str,
                       default=None,
                       help='path to sequence file')
    group.add_argument('-id', '--input_dir',
                       type=str,
                       default=None,
                       help='path to sequence directory')
    parser.add_argument('-o', '--output',
                        type=str,
                        default="results",
                        help='output directory (default: %(default)s)')
    parser.add_argument('-l', '--log',
                        type=str,
                        default="log",
                        help='log directory (default: %(default)s)')
    args = parser.parse_args()
    return args


def main(args):
    """
    Main function
    """
    # logging set up
    logging_path = Path(args.log)
    logging = Log(path=logging_path)
    logging.info(
        f"""Compressing {args.input}
            Path of log file: {logging_path}
            Output directory {args.output}""")

    ###################################################################
    DNAcom()


if __name__ == "__main__":
    args = arg_parse()
    main(args)
