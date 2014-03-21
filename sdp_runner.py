#!/usr/bin/env python2
from __future__ import division, print_function
import os
import argparse
import multiprocessing
from sdp import *


parser = argparse.ArgumentParser(description='Run sdps in parallel.')
parser.add_argument('directory', type=str,
                    help='output directory')
parser.add_argument('-s', '--spectrahedra', default=4, type=int,
                    help='number of spectrahedra to generate')
parser.add_argument('-o', '--objectives', default=10000, type=int,
                    help='number of objectives to evaluate')
args = parser.parse_args()

os.makedirs(args.directory)

def run_sdp(filename):
    with open(filename, 'w') as f:
        sdp = SDP()
        sdp.print_params(file=f)
        sdp.solve(args.objectives)
        sdp.print_results(file=f)

files = ['{0}/{1}.txt'.format(args.directory, i)
         for i in range(args.spectrahedra)]
pool = multiprocessing.Pool(maxtasksperchild=10)
pool.map(run_sdp, files)
