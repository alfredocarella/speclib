import itertools
import matplotlib.pyplot as plt
import numpy
import os
import pylab
from solverls.iterator import Iterator
from solverls.speclib import lagrange_interpolating_matrix, conj_grad_elem, conj_grad

__author__ = 'Alfredo Carella'


class LSProblem(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.residual = 10.0

        self.f, self.f_old = numpy.zeros(mesh.dof), numpy.zeros(mesh.dof)
        self.op_l, self.op_g, self.k_el, self.g_el = [], [], [], []

    def solve_linear(self):
        for el in self.mesh.elem:
            self.set_operators(el)
        self.set_boundary_conditions()
        self.f, cg_iterations = conj_grad_elem(self.k_el, self.g_el, self.mesh.gm, self.mesh.dof)
        self.residual = sum(self.compute_residual(el) for el in self.mesh.elem)

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

    def set_equations(self, el):
        raise NotImplementedError("Child classes must implement this method.")

    def set_boundary_conditions(self):
        raise NotImplementedError("Child classes must implement this method.")

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

    # The following should belong to another sub-class (e.g. LSProblemTimeMarching)
    def solve_linear_slab(self):
        self.f = numpy.zeros(self.mesh.dof)
        self.residual = 0
        for el in self.mesh.elem:
            self.set_operators(el)

            if el.number > 0:
                self.set_slab_boundary_conditions(el)
            else:
                self.set_boundary_conditions()

            f_elem, cg_iterations = conj_grad(self.k_el[0], self.g_el[0])
            self.f[el.nodes] = f_elem
            self.residual += self.compute_residual(el)

    def set_slab_boundary_conditions(self, el):
        for var in el.variables:
            f_index = el.nodes[el.pos[var]][0]
            gk_index = el.pos[var][0]
            self.k_el[0][gk_index, gk_index] += 1.0
            self.g_el[0][gk_index] += self.f[f_index]

    # The following should belong to another sub-class (e.g. LSProblemNonLinear)
    def solve_nonlinear(self):
        self.f = numpy.ones(self.mesh.dof)    # seed for guessing the solution
        it = Iterator(min_residual=1e-20, max_nonlinear_it=50, min_delta=1e-16)
        it.iterate(self)

        solver_report = "Iterations: {0!r:s}  -  Residual: {1:04.2e}  -  delta = {2:04.2e}"
        print(solver_report.format(it.solution_iterations, self.residual, it.delta))

    def solve_nonlinear_slab(self):
        raise NotImplementedError("This method has not been implemented yet.")  # TODO: Pending functionality


if __name__ == '__main__':

    from solverls.mesh1d import Mesh1D

    # TODO: This example must be added to the test suite.
    class TestLSProblemNonLinear(LSProblem):
        """Class for testing a poisson problem in 2 variables on N elements."""
        def __init__(self, mesh):
            LSProblem.__init__(self, mesh)
            self.solve_nonlinear()

        def set_equations(self, el):
            # Solution: f(x) = 2-(x-1)^2
            x = el.x_nv
            op_l = {'f.f': self.f[el.nodes] * el.dx}
            op_g = {'f': 2*x**3 - 6*x**2 + 2*x + 2}
            return op_l, op_g

        def set_boundary_conditions(self):
            weight, left_value = 1.0, 1.0
            self.k_el[0][0, 0] += weight
            self.g_el[0][0] += weight * left_value

    def minimum_nonlinear_example():
        """Testing iterative routine for solving a non-linear problem"""

        macro_grid, orders, variables = [0.0, 1.0, 2.0], [4, 4], ['f']
        my_mesh1d = Mesh1D(macro_grid, orders, variables)
        my_problem = TestLSProblemNonLinear(my_mesh1d)
        my_problem.plot()

        print('Test inputs:')
        print('macro_grid = %s' % my_mesh1d.macro_grid)
        print('orders = %s' % my_mesh1d.element_orders)
        print('variables = %s' % my_mesh1d.variables)
        print('')
        print('Test outputs:')
        print("The residual for this problem is %04.2e" % my_problem.residual)
    minimum_nonlinear_example()