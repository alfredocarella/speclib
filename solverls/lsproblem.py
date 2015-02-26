import itertools
import matplotlib.pyplot
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

    def compute_residual(self):
        """Compute the Least-Squares total residual."""

        list_of_elements = range(len(self.mesh.gm))

        residual = 0.0
        for el_ in list_of_elements:
            w = self.mesh.elem[el_].w_nv
            gm = self.mesh.elem[el_].nodes
            op_g = self.op_g[el_]
            op_l = self.op_l[el_]
            residual += w.dot((op_l.dot(self.f[gm])-op_g)**2)
        return residual

    def set_boundary_conditions(self):
        raise NotImplementedError("Child classes must implement this method.")

    def set_slab_boundary_conditions(self, elem):
        weight = 1.0
        for varName in self.mesh.list_of_variables:

            end_value_indices = self.mesh.gm[elem-1][self.mesh.pos[elem-1][varName]]
            start_value_indices = self.mesh.pos[elem][varName]
            f_index = end_value_indices[-1]
            gk_index = start_value_indices[0]

            self.k_el[0][gk_index, gk_index] += weight
            self.g_el[0][gk_index] += weight * self.f[f_index]

    def set_equations(self, el):
        raise NotImplementedError("Child classes must implement this method.")

    def set_operators(self):

        for el in self.mesh.elem:
            element_size = (el.order + 1) * len(el.variables)
            self.op_l.append(numpy.zeros((element_size, element_size)))
            self.op_g.append(numpy.zeros(element_size))
            opl_dict, opg_dict = self.set_equations(el)

            for (row, col) in itertools.product(self.mesh.variables, repeat=2):
                if (row+'.'+col) in opl_dict:
                    self.op_l[-1][numpy.ix_(el.pos[row], el.pos[col])] += opl_dict[row + '.' + col]
            for row in self.mesh.variables:
                if row in opg_dict:
                    self.op_g[-1][el.pos[row]] += opg_dict[row]

            lw_matrix = self.op_l[-1].T.dot(numpy.diag(el.x_nv))
            self.k_el.append(lw_matrix.dot(self.op_l[-1]))
            self.g_el.append(lw_matrix.dot(self.op_g[-1]))

    def set_solution(self, f):
        self.f_old = f

    def solve_linear(self):
        self.set_operators()
        self.set_boundary_conditions()
        self.f, cg_iterations = conj_grad_elem(self.k_el, self.g_el, self.mesh.gm, self.mesh.dof)

    def solve_nonlinear(self):
        self.f = numpy.ones(self.mesh.dof)    # or even --> self.mesh.x
        it = Iterator(min_residual=1e-20, max_nonlinear_it=50, min_delta=1e-16)
        it.iterate(self, self.set_operators, self.set_boundary_conditions, [0, 1])

        solver_report = "Iterations: {0!r:s}  -  Residual: {1:04.2e}  -  delta = {2:04.2e}"
        print(solver_report.format(it.number_of_iterations, self.residual, it.delta))

    def solve_linear_slab(self):
        self.f = numpy.zeros(self.mesh.dof_nv)

        self.set_operators([0])
        self.set_boundary_conditions()
        f_elem, cg_iterations = conj_grad(self.k_el[0], self.g_el[0])
        self.f[self.mesh.gm[0]] = f_elem

        for el_ in range(1, self.mesh.number_of_elements):
            self.set_operators([el_])
            self.set_slab_boundary_conditions(el_)
            f_elem, cg_iterations = conj_grad(self.k_el[0], self.g_el[0])
            self.f[self.mesh.gm[el_]] = f_elem

    # TODO: "lsproblem.plot_solution()" still needs to be polished and documented
    def plot_solution(self, list_of_variables=None, filename=None):

        if list_of_variables is None:
            list_of_variables = self.mesh.list_of_variables

        fig = matplotlib.pyplot.figure()
        for variable_name in list_of_variables:
            variable_number = self.mesh.variables.index(variable_name)
            x_in = numpy.zeros([])
            y_in = numpy.zeros([])
            x_out = numpy.zeros([])
            y_out = numpy.zeros([])
            for elem in range(len(self.mesh.elem)):
                variable_indices = self.mesh.gm[elem][self.mesh.elem[elem].pos[variable_name]]
                x_in_local = self.mesh.elem[elem].x_1v
                y_in_local = self.f[variable_indices]
                x_out_local = numpy.linspace(self.mesh.macro_grid[elem], self.mesh.macro_grid[elem+1], 10)
                y_out_local = lagrange_interpolating_matrix(x_in_local, x_out_local).dot(y_in_local)
                x_in = numpy.append(x_in, x_in_local)
                y_in = numpy.append(y_in, y_in_local)
                x_out = numpy.append(x_out, x_out_local)
                y_out = numpy.append(y_out, y_out_local)

            matplotlib.pyplot.subplot(100*len(list_of_variables) + 10 + variable_number+1)
            matplotlib.pyplot.plot(x_out, y_out, '--', linewidth=2.0)
            matplotlib.pyplot.plot(x_in, y_in, '.', markersize=8.0)
            matplotlib.pyplot.ylabel(variable_name)
            matplotlib.pyplot.xlabel('x (independent variable)')

        if filename is not None:
            if not os.path.exists('output'):
                os.makedirs('output')
            pylab.savefig('output//'+filename, bbox_inches=0)
            print("Functions %r have been printed to file '%s'" % (list_of_variables, filename))
        else:
            matplotlib.pyplot.show()

        return fig

    # TODO: After the other features are working
    def solve_nonlinear_slab(self):
        pass