import argparse
from logging import debug
from dna2vec import DNA2Vec
# from transformer import Transformer
from validation import Validation
from arithmeticCode import ArithmeticCoding
from pathlib import Path
from myLog import Log
# import transformer_kmer as trfrm
from transformer_kmer import build_transformer, ksize, continue_training
from utils import build_kmers
from kmerPredictor import KmerPredictor
from transformerEval import get_validation_metrics, evaluate_test_set
import transformerEval as te

def kmer_probs_to_base_probs(kmer_probs):
    """
    Transforms list of ("kmer", probability) tuples into list of [p(A), p(C), p(G), p(T)]
    """
    raise NotImplementedError

def iterate_arithmetic_encoder(base_probs, next_base):
    """
    Runs one iteration of the arithmetic encoder
    Args:
        base_probs (list):  [p(A), p(C), p(G), p(T)]
        next_base (str):    one of "A", "C", "G", or "T"
    """
    raise NotImplementedError

def iterate_arithmetic_decoder(base_probs, seq_num):
    """
    Runs one iteration of the arithmetic decoder, return the decoded base
    Args:
        base_probs (list):  [p(A), p(C), p(G), p(T)]
        seq_num:            Some representation of the encoded DNA sequence
    Returns:
        decoded_base (str): one of "A", "C", "G", or "T"
    """
    raise NotImplementedError

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

        self.transformer_path = "./checkpoints/len_16700/train"
        self.transformer = build_transformer()
        self.predictor = KmerPredictor(self.transformer)

        self.pipeline()
        # self.validation()

    def pipeline(self):
        """
        Pipeline of model
        """
        # embedding = DNA2Vec(self.input, self.model_path).build()  # returns None
        # embedding_path = Path("models/dna2vec")
        # dna_prob = Transformer(embedding_path, self.model_path)
        # ArithmeticCoding(self.input, dna_prob, self.output).encoding()

    def validation(self):
        """
        Decode the encoded sequence, benchmarking 
        """
        Validation(self.input, self.model)

    def compress_sequence(self, sequence):
        """
        Creates a compressed byte-representation of given DNA sequence
        """
        kmers = build_kmers(sequence, ksize)
        self.predictor.start_kmer_prediction()  # initialize prediction loop
        for kidx in range(len(kmers)):
            # Get kmer probabilities
            kmer_probs = self.predictor.get_next_kmer_probabilities()
            # Transform data and send through arithmetic encoding
            next_kmer = kmers[kidx]
            next_base = next_kmer[ksize - 1]
            # # The following two lines would deal with the arithmetic coder
            # base_probs = kmer_probs_to_base_probs(kmer_probs)
            # iterate_arithmetic_encoder(base_probs, next_base)
            # Feed next actual kmer into to the predictor model
            # print("Known output: ", self.predictor.output)
            print("Known output: ", self.predictor.detokenize_sequence(self.predictor.output))
            print("Top 5 Kmers: ", self.predictor.debug_get_n_highest_prob_kmers(kmer_probs, 5))
            print("Predicted: ", self.predictor.debug_get_highest_prob_kmer(kmer_probs))
            print("Actual: ", next_kmer)

            self.predictor.feedback_next_kmer(next_kmer)

    def decompress_sequence(self, seq_num, seq_len, initial_k1chunk):
        """
        Takes and decompresses the number representation of an encoded sequence 
        """
        next_kmer = "0" + initial_k1chunk # Add padding character to next kmer
        self.predictor.start_kmer_prediction()  # initialize prediction loop
        for kidx in range(seq_len):
            kmer_probs = self.predictor.get_next_kmer_probabilities()
            # # Transform probabilities and send through decoder
            base_probs = kmer_probs_to_base_probs(kmer_probs)
            next_base = iterate_arithmetic_decoder(base_probs, seq_num)
            next_kmer += next_base  # Append new base to kmer
            next_kmer = next_kmer[1:] # keep only bases after first base
            # Feed next actual kmer into to the predictor model
            self.predictor.feedback_next_kmer(next_kmer)



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
        

    # # Use the following to continue training on the transformer
    # # WARNING, this takes ~3hr/epoch and requires running on the largemem_q node
    # continue_training(num_epochs=4)
    
    # # Use this to run assessment of the test set:
    # # WARNING, it takes a while and requires largemem_q node
    # evaluate_test_set()

    # # Use this to run assessment of the validation set for a range of training epochs:
    # # WARNING, it takes a while and requires largemem_q node
    # get_validation_metrics(24, 27)


    # Use the following to print out a brief test of sequence predictions vs actual kmers
    test_sequence = "GATCACAGGTCTAAAAAAAA"
    compressor = DNACompressor("", "")
    compressor.compress_sequence(test_sequence)


if __name__ == "__main__":
    args = arg_parse()
    main(args)
