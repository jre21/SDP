#!/usr/bin/env python2
from __future__ import division, print_function

import argparse
import multiprocessing
import os
import random

from opt_utils import rand_matrix
from sdp import *


parser = argparse.ArgumentParser(description='Run sdps in parallel.')
parser.add_argument('directory', type=str,
                    help='output directory')
parser.add_argument('-s', '--spectrahedra', default=4, type=int,
                    help='number of spectrahedra to generate')
parser.add_argument('-o', '--objectives', default=10000, type=int,
                    help='number of objectives to evaluate')
parser.add_argument('-f', '--force_nodes', action='store_true',
                    help='force four nodes in symmetroid')
parser.add_argument('-n', '--negative', default=0, type=float,
                    help='probability for each eigenvalue to be negative '
                    + '(only effective if --force_nodes is used)')
parser.add_argument('--serialize', action='store_true',
                    help='disable parallelism for debug purposes')
args = parser.parse_args()

os.makedirs(args.directory)

def run_sdp(filename):
    A = B = C = D = None
    if args.force_nodes:
        def gen_matrix():
            """Generate a matrix of corank 2, such that each
            eigenvalue is negative with probability given by
            --negative 

            """
            vecs = rand_matrix(5,3,integer=True)
            signature = numpy.zeros((3,3), dtype=int)
            for i in range(len(signature)):
                if random.random() > args.negative:
                    signature[i,i] = -1
                else:
                    signature[i,i] = 1
            return vecs.dot(signature).dot(vecs.T)

        # Force nodes at (0,0,0) and (1,0,0) plus permutations by
        # giving A+D, B+D, C+D, and D corank 2.
        D = gen_matrix()
        A = gen_matrix() - D
        B = gen_matrix() - D
        C = gen_matrix() - D

    with open(filename, 'w') as f:
        sdp = SDP(A, B, C, D)
        sdp.print_params(file=f)
        sdp.solve(args.objectives)
        sdp.print_results(file=f)

files = ['{0}/{1}.txt'.format(args.directory, i)
         for i in range(args.spectrahedra)]
if args.serialize:
    for file in files:
        run_sdp(file)
else:
    pool = multiprocessing.Pool(maxtasksperchild=10)
    pool.map(run_sdp, files)
