import itertools
import os

from matplotlib import pyplot as plt
import numpy
import pylab

from solverls.speclib import lagrange_interpolating_matrix


class LSProblem:
    def __init__(self, mesh):
        self.mesh = mesh
        self.residual = 10.0

        self.f, self.f_old = numpy.zeros(mesh.dof), numpy.zeros(mesh.dof)
        self.op_l, self.op_g, self.k_el, self.g_el = [], [], [], []

    def set_operators(self, el):

        op_dict = self.set_equations(el)

        element_size = (el.order + 1) * len(el.variables)
        self.op_l.append(numpy.zeros((element_size, element_size)))
        for (row, col) in itertools.product(self.mesh.variables, repeat=2):
            if (row+'.'+col) in op_dict:
                self.op_l[-1][numpy.ix_(el.pos[row], el.pos[col])] += op_dict[row + '.' + col]
        self.op_g.append(numpy.zeros(element_size))
        for row in self.mesh.variables:
            if row in op_dict:
                self.op_g[-1][el.pos[row]] += op_dict[row]

        # lw_matrix = self.op_l[-1].T.dot(numpy.diag(el.w_nv))
        lw_matrix = self.op_l[-1].T * el.w_nv
        self.k_el.append(lw_matrix.dot(self.op_l[-1]))
        self.g_el.append(lw_matrix.dot(self.op_g[-1]))

    def compute_residual(self, el):
        """Compute the Least-Squares total residual."""
        w, gm = el.w_nv, el.nodes
        op_g = self.op_g[el.number]
        op_l = self.op_l[el.number]
        return w.dot((op_l.dot(self.f[gm])-op_g)**2)

    def plot(self, variables=None, filename=None):

        if variables is None:
            variables = self.mesh.variables

        fig = plt.figure()
        for var in variables:
            for el in self.mesh.elem:
                plt.subplot(100*len(variables) + 10 + el.variables.index(var)+1)
                plt.xlabel('x (independent variable)')
                plt.ylabel(var)

                x_in, y_in = el.x_1v, self.f[el.nodes[el.pos[var]]]
                plt.plot(x_in, y_in, '.', markersize=8.0, color='g')

                x_out = numpy.linspace(el.boundaries['x'][0], el.boundaries['x'][1], 20)
                y_out = lagrange_interpolating_matrix(x_in, x_out).dot(y_in)
                plt.plot(x_out, y_out, '-', linewidth=2.0, color='b')

        if filename is not None:
            if not os.path.exists('output'):
                os.makedirs('output')
            pylab.savefig('output//'+filename, bbox_inches=0)
            print("Functions %r have been printed to file '%s'" % (variables, filename))
        else:
            plt.show()

        return fig

    def set_equations(self, el):
        raise NotImplementedError("Child classes must implement this method.")

    def set_boundary_conditions(self):
        raise NotImplementedError("Child classes must implement this method.")

    def solve(self):
        raise NotImplementedError("Child classes must implement this method.")
