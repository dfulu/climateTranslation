import argparse
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, help='Path to the input file.')
parser.add_argument('--output_file', type=str, default='.', help="outputs path.")
parser.add_argument('--input_variable', type=str, help='')
parser.add_argument('--output_variables', type=str, help='', nargs='+')
opts = parser.parse_args()

with open(opts.input_file, 'r') as file:
    file = file.read()

new_file = '\n'.join([file.replace(f"/{opts.input_variable}/", f"/{v}/").replace(f"{opts.input_variable}_Aday", f"{v}_Aday") for v in opts.output_variables])

with open(opts.output_file, 'w') as file:
    file.write(new_file)