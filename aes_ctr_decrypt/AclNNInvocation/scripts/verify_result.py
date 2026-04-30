#!/usr/bin/python3
# coding=utf-8
import sys
import numpy as np

def verify_result(outputFile, goldenFile):
    outData = np.fromfile(outputFile, dtype=np.uint8)
    gdData  = np.fromfile(goldenFile, dtype=np.uint8)
    if outData.shape != gdData.shape:
        print("Size mismatch: output={}, golden={}".format(outData.shape, gdData.shape))
        return False

    diff = (outData != gdData).sum()
    if diff > 0:
        print("There are {} bytes different!".format(diff))
        return False
    return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 verify_result.py <cypher.bin> <golden.bin>")
        sys.exit(1)
    res = verify_result(sys.argv[1], sys.argv[2])
    if not res:
        print("Verification failed!")
        sys.exit(1)
    else:
        print("AES-CTR result matches golden!")
