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
parser.add_argument('--serialize', action='store_true',
                    help='disable parallelism for debug purposes')
args = parser.parse_args()

os.makedirs(args.directory)

def run_sdp(files):
    infile = files[0]
    outfile = files[1]
    sdp = SDP.from_file(infile)

    with open(outfile, 'w') as f:
        sdp.print_params(file=f)
        sdp.solve(args.objectives)
        sdp.print_results(file=f)

outfiles = ['{0}/{1}.txt'.format(args.directory, i)
            for i in range(len(args.infiles))]
if args.serialize:
    for files in zip(args.infiles, outfiles):
        run_sdp(files)
else:
    pool = multiprocessing.Pool(maxtasksperchild=10)
    pool.map(run_sdp, zip(args.infiles, outfiles))
