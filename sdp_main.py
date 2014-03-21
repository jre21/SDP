#!/usr/bin/env python2
from sdp import *

# run optimizations on a single spectrahedron, for testing purposes
import sys
if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 1000
sdp = SDP()
sdp.print_params()
sdp.solve(n)
sdp.print_results()
