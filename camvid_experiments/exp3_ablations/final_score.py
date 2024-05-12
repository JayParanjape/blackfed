import os
import sys
import numpy as np

file_path = sys.argv[1]
f = open(file=file_path,mode='r')
lines = f.readlines()
nums = []
for line in lines:
    line = line.strip()
    if 'Testing' in line:
        num = float(line[-6:])
        nums.append(num)

assert(len(nums)==16)
print(np.mean(nums))