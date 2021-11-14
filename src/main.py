import argparse
from dna2vec import DNA2Vec
# from transformer import Transformer
from validation import Validation
from arithmeticCode import ArithmeticCoding
from pathlib import Path
from myLog import Log
import transformer as trfrm


class DNACompressor():
    """
    DNA sequence compressor, the pipeline take the DNA sequence in fasta file, and output is the compressed binary file
    """

    def __init__(self, input, output):
        """
        Constructor

        Args:
            input (Path): path to input directory
            output (Path): path to output directory
        """
        self.input = input
        self.output = output
        self.model = None
        self.model_path = Path("models")
        Path.mkdir(self.model_path, parents=True, exist_ok=True)

        self.pipeline()
        # self.validation()

    def pipeline(self):
        """
        Pipeline of model
        """
        # embedding = DNA2Vec(self.input, self.model_path).build()  # returns None
        embedding_path = Path("models/dna2vec")
        dna_prob = Transformer(embedding_path, self.model_path)
        # ArithmeticCoding(self.input, dna_prob, self.output).encoding()

    def validation(self):
        """
        Decode the encoded sequence, benchmarking 
        """
        Validation(self.input, self.model)


def arg_parse():
    """
    Parsing arguments function
    """
    parser = argparse.ArgumentParser(
        description='DNA compression')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--input',
                       type=str,
                       default=None,
                       help='path to input directory')
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
        f'Compressing "{args.input}" directory, Path of log file: "{logging_path}",  Output directory "{args.output}"')

    ###################################################################
    # out_path = Path(args.output)
    # Path.mkdir(out_path, parents=True, exist_ok=True)

    # fasta_extension = [".fasta", ".fna", ".ffn", ".faa", ".frn", ".fa"]
    # files = [x for x in Path(args.input).iterdir()
    #          if x.is_file() and x.suffix in fasta_extension]
    # for file in files:
    #     logging.info(f"Compressing file {file}")
    #     DNACompressor(file, out_path)


    # ########################################################################
    # # Set Transformer hyperparameters
    # ########################################################################
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    



    


if __name__ == "__main__":
    args = arg_parse()
    main(args)
