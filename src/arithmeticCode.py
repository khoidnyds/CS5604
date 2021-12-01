import numpy as np
from pathlib import Path


class ArithmeticCoding():
    def __init__(self, input: str, prob: list, out: Path):
        self.input = input
        self.out = Path(out)
        self.nuc = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.inv_nuc = {v: k for k, v in self.nuc.items()}
        self.cdf = [sum(prob[0:x:1])
                    for x in range(5)]
        self.byte = ""
        self.tag = None
        self.e3 = False

    def bin2float(self, num: str):
        out = np.array([2**(-x-1) for x in range(len(num))])
        num = np.array([int(x) for x in num])
        return np.sum(np.matmul(out, num))

    def float2bin(self, num: str, point=False):
        if point:
            out = ['0', '.']
        else:
            out = []
        while not num.is_integer() and len(out) < 8:
            num *= 2
            if num >= 1:
                num -= 1
                out.append('1')
            else:
                out.append('0')
        return "".join(out)

    def encode_helper(self, high, low):
        while True:
            if high < 0.5:
                high *= 2
                low *= 2
                self.byte += "0"
            elif low > 0.5:
                high = 2*(high-0.5)
                low = 2*(low-0.5)
                self.byte += "1"
                if self.e3:
                    self.byte += "0"
                    self.e3 = False
            # elif high-low < 0.5 and low < 0.5 and high > 0.5:
            #     high = 2*(high-0.25)
            #     low = 2*(low-0.25)
            #     self.e3 = True
            else:
                return high, low

    def decode_helper(self, high, low, bytes):
        if high < 0.5:
            high *= 2
            low *= 2
            bytes = bytes[1:]
        elif low > 0.5:
            high = 2*(high-0.5)
            low = 2*(low-0.5)
            bytes = bytes[1:]
        elif high-low < 0.5 and low < 0.5 and high > 0.5:
            high = 2*(high-0.25)
            low = 2*(low-0.25)
        return high, low, bytes

    def encoding(self):
        # TODO: Byte stream of metadata

        low, high = 0.0, 1.0
        # Iterate through the word to find the final range.
        for c in range(len(self.input)):
            new_low = low + (high-low)*self.cdf[self.nuc[self.input[c]]]
            new_high = low + (high-low)*self.cdf[self.nuc[self.input[c]]+1]
            high, low = self.encode_helper(new_high, new_low)

        self.byte += '1'
        # self.byte += self.float2bin((high+low)/2, point=False)

    def decoding(self, bytes, cdf):
        cdf = np.array(cdf)
        new_cdf = cdf
        decoded = []
        low = 0
        high = 1
        while True:
            val = self.bin2float(bytes)
            idx = np.argmax(new_cdf > val)-1
            decoded.append(self.inv_nuc[idx])
            if len(decoded) == len(self.input):
                return "".join(decoded)

            new_low = low + (high-low)*self.cdf[idx]
            new_high = low + (high-low)*self.cdf[idx+1]
            high, low, bytes = self.decode_helper(new_high, new_low, bytes)
            new_cdf = low + (high-low)*cdf


seq = "CGATTATAT"
a = ArithmeticCoding(
    seq, [0.2, 0.3, 0.2, 0.3], "results/example.bin")
a.encoding()
print(a.decoding(a.byte, a.cdf))
