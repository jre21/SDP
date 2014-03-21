#!/usr/bin/env python2
from __future__ import division, print_function
import os
import argparse
import multiprocessing
from sdp import *


parser = argparse.ArgumentParser(
    description='Reprocess sdps which have already been run.'
)
parser.add_argument('directory', type=str,
                    help='output directory')
parser.add_argument('infiles', metavar='file', nargs='+',
                    help='files to reprocess')
parser.add_argument('-o', '--objectives', default=10000, type=int,
                    help='number of objectives to evaluate')
args = parser.parse_args()

os.makedirs(args.directory)

def run_sdp(files):
    infile = files[0]
    outfile = files[1]

    with open(infile) as f:
        lines = f.readlines()

    A = numpy.array(eval(lines[6])).reshape(5,5)
    B = numpy.array(eval(lines[13])).reshape(5,5)
    C = numpy.array(eval(lines[20])).reshape(5,5)
    sdp = SDP(A,B,C)

    with open(outfile, 'w') as f:
        sdp.print_params(file=f)
        sdp.solve(args.objectives)
        sdp.print_results(file=f)

outfiles = ['{0}/{1}.txt'.format(args.directory, i)
            for i in range(len(args.infiles))]
pool = multiprocessing.Pool(maxtasksperchild=10)
pool.map(run_sdp, zip(args.infiles, outfiles))
