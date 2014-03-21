#!/usr/bin/env python2
from __future__ import division, print_function

import fileinput;

# collect and display aggregate statistics from a set of output files
nodes = [[{'count':0, 'probs':[]} for j in range(21)] for i in range(21)]
spec = 'spectrahedral nodes: '
sym = 'symmetroid nodes: '
spec_node = 'node '
sym_node = 'symmetroid node '
prob = 'probability: '
probs = [0]
for line in fileinput.input():
    if fileinput.isfirstline():
        if max(probs) != 0:
            nodes[sym_nodes][spec_nodes]['probs'].append(probs[1:])
            probs = [0]
    elif line.startswith(spec):
        sym_nodes = int(line[len(spec):])
    elif line.startswith(sym):
        spec_nodes = int(line[len(sym):])
        nodes[sym_nodes][spec_nodes]['count'] += 1
    elif line.startswith(spec_node):
        cur_node = int(line[len(spec_node):line.index(':')])
    elif line.startswith(prob):
        probs.append(float(line[len(prob):]))
    elif line.startswith(sym_node):
        fileinput.nextfile()
if max(probs) != 0:
    nodes[sym_nodes][spec_nodes]['probs'].append(probs[1:])
    probs = [0]

if max(probs) != 0:
    nodes[sym_nodes][spec_nodes]['probs'].append(probs[1:])
    probs = [0]

divisor = sum([i['count'] for j in nodes for i in j])
tuples = []
node_total = 0
highest_node = 0
node_count = 0
for i in range(len(nodes)):
    for j in range(len(nodes[i])):
        if nodes[i][j]['count'] != 0:
            next_tuple = {
                'rho': i, 'sigma': j,
                'probability': round(nodes[i][j]['count']/divisor,5)
            }
            count = len(nodes[i][j]['probs'])
            if count:
                next_tuple['node total'] = round(
                    sum([sum(x) for x in nodes[i][j]['probs']])/count
                    , 4)
                node_total += count * next_tuple['node total']
                next_tuple['highest node'] = round(
                    sum([max(x) for x in nodes[i][j]['probs']])/count
                    , 4)
                highest_node += count * next_tuple['highest node']
                node_count += count
            tuples.append(next_tuple)

tuples.sort(key=lambda x: 100*x['rho']+x['sigma'])

for tuple in tuples:
    print('{0}, {1}: {2}, {3}'.format(
        tuple.pop('rho'),
        tuple.pop('sigma'),
        tuple.pop('probability'),
        str(tuple)[1:-1]
    ))
if node_count:
    print('highest node: {0}, node total: {1}'.format(
        highest_node/node_count, node_total/node_count
    ))
