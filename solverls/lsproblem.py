import itertools
import os

from matplotlib import pyplot as plt
import numpy
import pylab

from solverls.spectral import interpolant_evaluation_matrix


class LSProblem:
    def __init__(self, mesh=None):
        self.residual = 100.0

        if mesh:
            self.mesh = mesh
            self.f, self.f_old = numpy.zeros(mesh.dof), numpy.zeros(mesh.dof)
            self.op_l = [None for _ in range(len(self.mesh.gm))]
            self.op_g = [None for _ in range(len(self.mesh.gm))]
            self.k_el = [None for _ in range(len(self.mesh.gm))]
            self.g_el = [None for _ in range(len(self.mesh.gm))]

    def set_problem(self):
        for element in self.mesh.elem:
            self.set_operators(element)
        self.set_boundary_conditions()

    def set_operators(self, el):

        op_dict = self.define_equations(el)

        element_size = (el.order + 1) * len(el.variables)
        self.op_l[el.number] = numpy.zeros((element_size, element_size))
        for (row, col) in itertools.product(self.mesh.variables, repeat=2):
            if (row+'.'+col) in op_dict:
                self.op_l[el.number][numpy.ix_(el.pos[row], el.pos[col])] += op_dict[row + '.' + col]

        self.op_g[el.number] = numpy.zeros(element_size)
        for row in self.mesh.variables:
            if row in op_dict:
                self.op_g[el.number][el.pos[row]] += op_dict[row]

        lw_matrix = el.w_nv * self.op_l[el.number].T  # faster than "self.op_l[el.number].T.dot(numpy.diag(el.w_nv))"
        self.k_el[el.number] = lw_matrix.dot(self.op_l[el.number])
        self.g_el[el.number] = lw_matrix.dot(self.op_g[el.number])

    def compute_residual(self, el):
        """Compute the Least-Squares total residual."""
        w, gm = el.w_nv, self.mesh.gm[el.number]
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

                x_in, y_in = el.x_1v, self.f[self.mesh.gm[el.number][el.pos[var]]]
                plt.plot(x_in, y_in, '.', markersize=8.0, color='g')

                x_out = numpy.linspace(*el.boundaries['x'], num=20)
                y_out = interpolant_evaluation_matrix(x_in, x_out).dot(y_in)
                plt.plot(x_out, y_out, '-', linewidth=2.0, color='b')

        if filename is not None:
            if not os.path.exists('output'):
                os.makedirs('output')
            pylab.savefig('output//'+filename, bbox_inches=0)
            print("Functions %r have been printed to file '%s'" % (variables, filename))
        else:
            plt.show()

        return fig

    def define_equations(self, el):
        raise NotImplementedError("Method not implemented in child class.")

    def set_boundary_conditions(self):
        raise NotImplementedError("Method not implemented in child class.")

    def solve(self):
        raise NotImplementedError("Method not implemented in child class.")
