import os
import sys
import subprocess

x = open('RECORDS.txt', 'r').read().splitlines()
x = x[64:]
for i in x:
	subprocess.call([sys.executable, 'data_chunking.py', i])
	print(i, "DONE")