import numpy as np


def bin2float(num: str):
    out = np.array([2**(-x-1) for x in range(len(num))])
    num = np.array([int(x) for x in num])
    return np.sum(np.matmul(out, num))


print(bin2float('110001'))
