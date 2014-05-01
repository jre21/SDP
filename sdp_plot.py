#!/usr/bin/env python2
from __future__ import division, print_function
import argparse
from sdp import *

parser = argparse.ArgumentParser(
    description='Plot an sdp.'
)
parser.add_argument('file', type=str,
                    help='spectrahedron to plot')
parser.add_argument('-t', '--theta', default=10, type=int,
                    help='number of samples along theta axis')
parser.add_argument('-p', '--phi', default=20, type=int,
                    help='number of samples along phi')
args = parser.parse_args()
sdp = SDP.from_file(args.file)
sdp.plot(args.theta, args.phi)
