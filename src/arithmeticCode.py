from decimal import Decimal
import numpy as np
from utils import build_kmers
from pathlib import Path
import pickle
from pyfaidx import Fasta


class ArithmeticCoding():
    def __init__(self, input, kmer_size, model, out):
        self.kmer_size = kmer_size
        self.input = input
        self.model = model
        self.nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.words = build_kmers(self.input, self.kmer_size)
        self.prob_table = {k: self.model(k) for k in self.words}
        self.length = len(self.words)
        self.dont_know_how_to_encode = input[:kmer_size-1]
        self.out = Path(out)

    def get_range(self, min_val, max_val, prob):
        scale_factor = 1/(max_val-min_val)
        range_prob = np.array([sum(prob[0:x:1])
                              for x in range(0, len(prob+1))])
        out = range_prob/scale_factor+min_val
        out_range = {k: [out[k], out[k+1]] for k in range(3)}
        out_range[3] = [out[-1], max_val]
        return out_range

    def check_range(self, prob, val):
        for idx, (min_val, max_val) in prob.items():
            if val > min_val and val < max_val:
                return idx

    def encoding(self):
        min_range = Decimal(0)
        max_range = Decimal(1)
        for word in self.words:
            nuc = self.nuc_map[word[-1]]
            prob = self.get_range(min_range, max_range, self.prob_table[word])
            min_range, max_range = prob[nuc]
        self.encoded = (min_range+max_range)/2
        return self.encoded

    def decoding(self):
        decoded = []
        min_range = Decimal(0)
        max_range = Decimal(1)
        for i in range(self.length):
            prob = self.get_range(min_range, max_range,
                                  self.prob_table[self.words[i]])
            char = self.check_range(prob, self.encoded)
            decoded.append(list(self.nuc_map.keys())[
                           list(self.nuc_map.values()).index(char)])
            min_range, max_range = prob[char]
        self.decoded = self.dont_know_how_to_encode + "".join(decoded)
        return self.decoded

    def write_to_file(self):
        pickle.dump(self.encoded, open(self.out, "wb"))

    # def read_from_file(self):
    #     with open(self.out, "rb") as file:
    #         file.read(bytearray(self.decoded))


def random_out(input):
    nums = np.random.uniform(size=4)
    nums = np.array([Decimal(num) for num in nums])
    return nums/sum(nums)


seq = "CGTAGCTGACGTCGATGCTATCGATCGTAC"
a = ArithmeticCoding(
    seq, 13, random_out, "results/example.bin")
a.encoding()
a.decoding()
a.write_to_file()
assert seq == a.decoded
