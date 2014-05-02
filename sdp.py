from __future__ import division, print_function

import contextlib
import os
import random
import shutil
import subprocess
import tempfile

import numpy
import scipy.linalg as linalg
import sympy

from opt_utils import rand_matrix


class SDP:
    """Class to generate and test SDP problems.  Normally, only methods
    under 'main components' heading should be called from external
    code.

    """
    def __init__(self, A=None, B=None, C=None, D=None):
        """Generate internal variables.

        The spectrahedron is the surface det(xA + yB + zC + D) = 0
        All points are represented as lists of floats
        A, B, C, D: represented as numpy ndarrays
        matrices: [A, B, C, D], collected for convenience
        mins: list of minimizing points of randomly-generated SDPs
        pmins: list indicating points with multiplicities
            Each element of pmins takes the form
            [location, occurances, eigenvalues]
        nodes: list of spectrahedral nodes, in the form
            [location, fractional occurances, eigenvalues].
            Nodes are sorted in descending order by frequency.
        spec_nodes: nodes on surface of spectrahedron
        sym_nodes: other real nodes on symmetroid
        complex_nodes: nodes with nonzero imaginary parts
            all nodes represented as [location, eigenvalues]
        total_nodes: total number of nodes (including complex)
        trials: number of calls to cvx
        psd_spec: whether spectrahedron contains psd component
        nsd_spec: ditto for nsd
        fully_bounded_directions (fully_unbounded_directions):
            number of directions in which optimizations on both
            spectrahedra are bounded (unbounded)

        """
        self.mins = []
        self.pmins = []
        self.nodes = []
        self.spec_nodes = []
        self.sym_nodes = []
        self.complex_nodes = []
        self.total_nodes = 0
        self.trials = 0
        self.psd_spec = True
        self.nsd_spec = True
        self.fully_bounded_directions = 0
        self.fully_unbounded_directions = 0

        if A is None:
            self.A = rand_matrix(5,5,symmetric=True,integer=True)
        else:
            self.A = A

        if B is None:
            self.B = rand_matrix(5,5,symmetric=True,integer=True)
        else:
            self.B = B

        if C is None:
            self.C = rand_matrix(5,5,symmetric=True,integer=True)
        else:
            self.C = C

        if D is None:
            self.D = numpy.identity(5, dtype=int)
        else:
            self.D = D

        self.matrices = [self.A, self.B, self.C, self.D]


    @classmethod
    def from_file(cls, filename):
        """Initialize an instance from an output file."""
        with open(filename) as f:
            lines = f.readlines()

        A = numpy.array(eval(lines[6])).reshape(5,5)
        B = numpy.array(eval(lines[13])).reshape(5,5)
        C = numpy.array(eval(lines[20])).reshape(5,5)
        D = numpy.array(eval(lines[27])).reshape(5,5)
        return cls(A,B,C,D)


    #
    # Utility functions
    #
    def matrix(self, vector):
        """Return (xA+yB+zC+D) at the point designated by vector."""
        vec = vector[:]
        vec.append(1)
        return sum([vec[i] * self.matrices[i]
                                     for i in range(len(self.matrices))])


    def eigenvalues(self, vector):
        """Return the eigenvalues of (xA+yB+zC+D) at a point."""
        svd = linalg.svd(self.matrix(vector))
        eivals = svd[1]
        for i in range(len(eivals)):
            if svd[0][i,i] * svd[2][i,i] < 0:
                eivals[i] *= -1
        return eivals


    def cvx_solve(self, obj, verbose=False):
        """Solve an objective function with cvx.

        obj: vector representing the linear objective to be minimized
        verbose: whether cvx should print verbose output to stdout

        Returns a pair representing the optimum of the psd and nsd
        components (None if that component is either infeasible or
        unbounded along the optimization direction).  Also sets
        psd_spec, nsd_spec, and fully_(un)bounded_directions.

        """
        import cvxpy as cvx

        x = cvx.Variable(name='x')
        y = cvx.Variable(name='y')
        z = cvx.Variable(name='z')
        # dummy variable to code semidefinite constraint
        T = cvx.semidefinite(5,name='T')
        spec = self.A * x + self.B * y + self.C * z + self.D
        objective = cvx.Minimize(obj[0]*x + obj[1]*y + obj[2]*z)
        out_psd = out_nsd = None

        # check PSD component
        if self.psd_spec:
            psd = cvx.Problem(objective, [T == spec])
            psd.solve(verbose=verbose)
            if psd.status == cvx.OPTIMAL:
                out_psd = [x.value, y.value, z.value]
            elif psd.status == cvx.INFEASIBLE:
                self.psd_spec = False

        # check NSD component
        if self.nsd_spec:
            nsd = cvx.Problem(objective, [T == -spec])
            nsd.solve(verbose=verbose)
            if nsd.status == cvx.OPTIMAL:
                out_nsd = [x.value, y.value, z.value]
                if self.psd_spec and psd.status == cvx.OPTIMAL:
                    self.fully_bounded_directions += 1
            elif nsd.status == cvx.UNBOUNDED \
               and self.psd_spec and psd.status == cvx.UNBOUNDED:
                self.fully_unbounded_directions += 1
            elif nsd.status == cvx.INFEASIBLE:
                self.nsd_spec = False

        return out_psd, out_nsd


    #
    # functions for singular handler
    #
    def get_nodes_from_singular(self):
        """Determine location of nodes with singular.

        Returns list of real nodes.

        """
        with tempfile.NamedTemporaryFile() as f:
            self.print_singular_script(file=f)
            f.flush()
            output = subprocess.check_output(['singular',f.name])
        return self.parse_singular_output(output)


    def matrix_to_singular(self, matrix):
        """Format a matrix for input into singular.

        matrix: matrix to format

        Returns string usable in singular script.

        """
        # Singular expects matrices in a flattened form without any
        # decoration.  E.g., a 2x2 matrix would be initialized by
        ## matrix m[2][2] = m11, m12, m21, m22;
        return str([i for i in matrix.flat])[1:-1]


    def print_singular_script(self, template="data/singular_script",
                              file=None):
        """Create a singular script suitable for execution from template.

        template: template file from which to generate script
        file: where to write out script

        """
        with open(template) as f:
            for line in f.readlines():
                print(line.format(A=self.matrix_to_singular(self.A),
                                  B=self.matrix_to_singular(self.B),
                                  C=self.matrix_to_singular(self.C),
                                  D=self.matrix_to_singular(self.D)),
                      end='',file=file)


    def parse_singular_output(self, string):
        """Parse the output from singular and return list of nodes.

        string: raw output from singular call

        Returns list of real nodes and sets total_nodes.

        """
        # Singular uses a tree-like structure for its output, displaying
        # solution n as
        ## [n]:
        ##    [1]:
        ##       var1
        ##    [2]:
        ##       var2
        ##    [3]:
        ##       var3
        # where vari is the value of the i'th variable.  Complex numbers
        # are represented in the format
        ## (re+i*im)
        split = string[string.find('[1]'):].splitlines()
        vectors = []
        for i in range(0,len(split),7):
            self.total_nodes += 1
            if '(' in split[i+2] or '(' in split[i+4] or '(' in split[i+6]:
                continue
            vectors.append([float(split[i+j]) for j in range(2,8,2)])
        return vectors


    #
    # functions for bertini handler
    #
    def get_nodes_from_bertini(self, verbose=False):
        """Determine location of nodes with bertini.

        verbose: set to True to print Bertini output to stdout
                 default is false (suppress Bertini output)

        Returns list of all nodes.

        """
        @contextlib.contextmanager
        def temp_directory():
            tmpdir = tempfile.mkdtemp()
            yield tmpdir
            shutil.rmtree(tmpdir)

        with temp_directory() as tmpdir:
            self.print_bertini_script(tmpdir)
            cwd = os.getcwd()
            os.chdir(tmpdir)
            if verbose:
                subprocess.call(['bertini'])
            else:
                with open(os.devnull) as null:
                    subprocess.call(['bertini'], stdout=null, stderr=null)
            os.chdir(cwd)
            retval = self.parse_bertini_output(tmpdir)
        return retval


    def print_bertini_script(self, directory, template="data/bertini_input"):
        """Create a bertini script suitable for execution from template.

        directory: working directory for bertini
        template: template file from which to generate script

        """
        x, y, z = sympy.symbols('x y z')
        det = sympy.det(
            x * sympy.Matrix(self.A) + y * sympy.Matrix(self.B)
            + z * sympy.Matrix(self.C) + sympy.Matrix(self.D)
        )
        difx = sympy.diff(det,x)
        dify = sympy.diff(det,y)
        difz = sympy.diff(det,z)

        with open(template) as inp:
            with open(directory + '/input', mode='w') as out:
                for line in inp.readlines():
                    print(line.format(
                        F=str(det).replace('**','^'),
                        G=str(difx).replace('**','^'),
                        H=str(dify).replace('**','^'), 
                        I=str(difz).replace('**','^')
                    ), file=out, end='')


    def parse_bertini_output(self, directory):
        """Parse output from bertini and return list of nodes.

        directory: working directory for bertini

        Returns list of all nodes and sets total_nodes.

        """
        with open(directory + '/finite_solutions') as f:
            lines = f.readlines()

        # The file finite_solutions contains the number of solutions,
        # followed by a blank line, followed by the solutions.  The
        # value of each variable is listed on its own line in the form
        # 're im', and solutions are separated by blank lines.
        # Graphically this is:
        #
        ## n_solutions
        ##
        ## x1.re x1.im
        ## y1.re y1.im
        ## z1.re z1.im
        ##
        ## x2, etc ...

        # trim trailing newlines
        while lines[-1] == '\n':
            lines = lines[:-1]

        # list of vectors of complex numbers represented as [re, im]
        complex_vecs = []
        for line in lines:
            if line == '\n':
                complex_vecs.append([])
                continue
            line = line[:-1].split()
            if len(line) > 1:
                complex_vecs[-1].append([float(x) for x in line])
            else:
                self.total_nodes = int(line[0])

        # detect real vectors, and convert complex ones to native format
        vecs = []
        for vec in complex_vecs:
            re, im = list(zip(*vec))
            # see if we've found the origin
            if max(re) < 1e-5 and max(im) < 1e-5:
                vecs.append([0 for i in range(len(re))])
            # use min() to compactly express a conjunction
            elif min([abs(im[i]) <= 1e-5 * abs(re[i]) for i in range(len(re))]):
                vecs.append(list(re))
            else:
                vecs.append([complex(v[0],v[1]) for v in vec])

        return vecs


    #
    # main components
    #
    def print_params(self, file=None):
        """Print the matrix parameters.

        file: file to print to (default: stdout)

        """
        print('A:', file=file)
        print(self.A, file=file)
        print([a for a in self.A.flat], file=file)
        print('B:', file=file)
        print(self.B, file=file)
        print([b for b in self.B.flat], file=file)
        print('C:', file=file)
        print(self.C, file=file)
        print([c for c in self.C.flat], file=file)
        print('D:', file=file)
        print(self.D, file=file)
        print([d for d in self.D.flat], file=file)
        print('', file=file)


    def solve(self, n=1, verbose=False):
        """Solve optimization problems.

        n: number of optimizations
        verbose: whether cvx should print verbose output to stdout

        Appends results to mins, plus additional side effects
        described in cvx_solve().

        """
        for i in range(n):
            c, = rand_matrix(1,3)
            psd, nsd = self.cvx_solve(c, verbose)
            if psd is not None:
                self.mins.append(psd)
            if nsd is not None:
                self.mins.append(nsd)

        self.trials += n


    def plot(self, ntheta=10, nphi=20, verbose=False):
        """Generate a plot of the spectrahedron.

        Sampling is done using objectives evenly spaced in spherical
        coordinates, with ntheta and nphi controlling the number of
        subdivisions along the respective axis.

        The resulting plot is displayed interactively.  Also invokes
        side effects of cvx_solve().

        """
        from mayavi import mlab

        dphi = 2*numpy.pi/nphi
        dtheta = numpy.pi/ntheta
        phi,theta = numpy.mgrid[0:numpy.pi+dphi*1.5:dphi,
                                0:2*numpy.pi+dtheta*1.5:dtheta]
        Xp = numpy.zeros_like(theta)
        Yp = numpy.zeros_like(theta)
        Zp = numpy.zeros_like(theta)
        Xn = numpy.zeros_like(theta)
        Yn = numpy.zeros_like(theta)
        Zn = numpy.zeros_like(theta)
        for i in range(len(phi)):
            for j in range(len(phi[i])):
                obj = [numpy.cos(phi[i,j])*numpy.sin(theta[i,j]),
                       numpy.sin(phi[i,j])*numpy.sin(theta[i,j]),
                       numpy.cos(theta[i,j])]
                psd, nsd = self.cvx_solve(obj, verbose)
                if psd is not None:
                    Xp[i,j], Yp[i,j], Zp[i,j] = psd
                if nsd is not None:
                    Xn[i,j], Yn[i,j], Zn[i,j] = nsd

        if self.psd_spec:
            mlab.mesh(Xp, Yp, Zp, colormap='Greys')
        if self.nsd_spec:
            mlab.mesh(Xn, Yn, Zn, colormap='Greys')
        mlab.axes()
        mlab.show()


    def get_nodes(self, handler=None):
        """Determine location of real nodes, and classify them.

        handler() must output nodes as lists of points.

        Sets spec_nodes, sym_nodes, and complex_nodes.

        """
        if handler is None:
            handler = self.get_nodes_from_bertini
        for vector in handler():
            e = self.eigenvalues(vector)
            # use min to compactly express a conjunction
            if min([v.conjugate() == v for v in vector]):
                if min([v >= 0 for v in e[:-2]]) \
                   or min([v <= 0 for v in e[:-2]]):
                    self.spec_nodes.append([vector,e])
                else:
                    self.sym_nodes.append([vector,e])
            else:
                self.complex_nodes.append([vector,e])

        # define a canonical ordering on nodes for ease of comparison
        self.spec_nodes.sort(key = lambda x: x[0][0])
        self.sym_nodes.sort(key = lambda x: x[0][0])
        self.complex_nodes.sort(key = lambda x: x[0][0].real)


    def process(self, tolerance=1e-3):
        """Process minima to determine number of occurances.

        Points x and y are considered identical if
        norm(x-y)/max(norm(z)) is less than tolerance, where the
        maximum is over locations of spectrahedral nodes.

        Calls get_nodes if necessary, sets self.pmins, and clears
        self.mins.

        """
        if not self.total_nodes:
            self.get_nodes()
        if self.spec_nodes:
            if not self.pmins:
                self.pmins = [[node[0], 0, node[1]]
                              for node in self.spec_nodes]
            maxdelta = tolerance * max([linalg.norm(y[0]) for y in self.pmins])
            for y in self.pmins:
                yy = numpy.array(y[0])
                for x in self.mins:
                    delta = linalg.norm(numpy.array(x)-yy)
                    if delta <= maxdelta:
                        y[1] += 1
        # zero out mins once all elements are processed
        self.mins = []


    def gen_nodes(self):
        """Fetch all nodes with percent of minima occuring at each.

        Calls process if necessary, and sets self.nodes.

        threshold: minimum number of points to be considered a node.
        If |x-y|/|x| < rel_threshold, discard whichever of x and y has
        fewer points.

        """
        if self.mins != [] or not self.total_nodes:
            self.process()

        self.nodes = []
        if self.trials is not 0:
            for i in self.pmins:
                self.nodes.append([i[0], i[1] / self.trials, i[2]])
            self.nodes.sort(key=lambda x: x[1], reverse=True)
        else:
            for i in self.pmins:
                self.nodes.append([i[0], 0, i[2]])
         

    def print_results(self, file=None):
        """Print results of optimization to file (default: stdout)."""
        if self.nodes == []:
            self.gen_nodes()
        print("spectrahedral nodes: {0}".format(len(self.pmins)), file=file)
        print("symmetroid nodes: {0}".format(
            len(self.sym_nodes) + len(self.pmins)
        ), file=file)
        print("total nodes: {0}".format(self.total_nodes), file=file)

        # Flag this file if any computed node is not a double root
        # of the determinant polynomial.
        invalid_node = False
        for node in self.spec_nodes:
            if node[1][3]/node[1][2] > 1e-5:
                invalid_node = True
        for node in self.sym_nodes:
            if node[1][3]/node[1][2] > 1e-5:
                invalid_node = True
        if invalid_node:
            print("invalid node detected", file=file)

        print("", file=file)

        if self.trials is not 0:
            print("has psd component: {0}".format(self.psd_spec), file=file)
            print("has nsd component: {0}".format(self.nsd_spec), file=file)
            if self.psd_spec and self.nsd_spec:
                print("fraction of twice-solvable objectives: {0}".format(
                    self.fully_bounded_directions / self.trials
                ), file=file)
                print("fraction of twice-unbounded objectives: {0}".format(
                    self.fully_unbounded_directions / self.trials
                ), file=file)
            print("", file=file)

        for i in range(len(self.nodes)):
            print("node {0}:".format(i+1), file=file)
            print("location: {0}".format(self.nodes[i][0]), file=file)
            if self.trials is not 0:
                print("probability: {0}".format(self.nodes[i][1]), file=file)
            print("eigenvalues:", file=file)
            print(self.nodes[i][2], file=file)
            print('', file=file)
        for i in range(len(self.sym_nodes)):
            print("symmetroid node {0}:".format(i+1), file=file)
            print("location: {0}".format(self.sym_nodes[i][0]), file=file)
            print("eigenvalues:", file=file)
            print(self.sym_nodes[i][1], file=file)
            print("", file=file)
        for i in range(len(self.complex_nodes)):
            print("complex node {0}:".format(i+1), file=file)
            print("location: {0}".format(self.complex_nodes[i][0]), file=file)
            print("eigenvalues:", file=file)
            print(self.complex_nodes[i][1], file=file)
            print("", file=file)
