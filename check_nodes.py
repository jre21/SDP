import sys
from sdp import *

# Re-check nodes for a single sdp.  Useful for debugging new code paths.

with open(sys.argv[1]) as f:
    lines = f.readlines()

A = numpy.array(eval(lines[6])).reshape(5,5)
B = numpy.array(eval(lines[13])).reshape(5,5)
C = numpy.array(eval(lines[20])).reshape(5,5)

sdp = SDP(A,B,C)
print(sdp.get_nodes())
