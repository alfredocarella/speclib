import itertools
import math

from nose.tools import assert_almost_equal
import numpy

from solverls.lsproblemlinear import LSProblemLinear
from solverls.lsproblemtimeslab import LSProblemTimeSlab
from solverls.lsproblemnonlinear import LSProblemNonLinear
from solverls.mesh1d import Mesh1D

__author__ = 'Alfredo Carella'


class TestLSProblemLinear1El1Var(LSProblemLinear):
    """Class for testing a simple Poisson problem in 1 variable on 1 element."""

    def define_equations(self, el):
        # Solution: f(x) = 0.5 * (x^2 - (x0+x1)*x + x0*x1) + 3
        op_dict = {'f.f': el.dx.dot(el.dx),
                   'f': -1.0 * numpy.ones(el.order + 1)}
        return op_dict

    def set_boundary_conditions(self):
        weight = 1.0
        left_value = 3.0
        right_value = 3.0
        order = self.mesh.elem[-1].order
        self.k_el[0][0, 0] += weight
        self.g_el[0][0] += weight * left_value
        self.k_el[-1][order, order] += weight
        self.g_el[-1][order] += weight * right_value

    def get_analytical_solution(self):
        x0 = self.mesh.elem[0].boundaries['x'][0]
        x1 = self.mesh.elem[-1].boundaries['x'][1]
        x = x0
        for el in self.mesh.elem:
            x = numpy.append(x, el.x_1v[1:])
        analytical_solution = 0.5 * (-1 * x ** 2 + (x0 + x1) * x - x0 * x1) + 3
        return analytical_solution

    def test_mesh_generator(self):
        macro_grids = [[0, 2], [-1, 4]]
        orders = [4, 7]
        for macro_grid, order in itertools.product(macro_grids, orders):
            yield self.check_residual_and_solution, Mesh1D(macro_grid, order, ['f'])

    def check_residual_and_solution(self, prob_mesh):
        self.__init__(prob_mesh)
        self.solve()
        assert_almost_equal(self.residual, 0.0)
        numpy.testing.assert_allclose(self.f[0:prob_mesh.dof_1v], *[self.get_analytical_solution()])


class TestLSProblemLinearNEl2Var(TestLSProblemLinear1El1Var):
    """Class for testing a simple Poisson problem in 2 variables on N elements."""

    def define_equations(self, el):
        # Solution: f(x) = 0.5 * (x^2 - (x0+x1)*x + x0*x1) + 3
        op_dict = {'f.f': el.dx,
                   'f.g': -1.0 * numpy.identity(el.order + 1),
                   'f': numpy.zeros(el.order + 1),

                   'g.f': numpy.zeros((el.order + 1, el.order + 1)),
                   'g.g': el.dx,
                   'g': -1.0 * numpy.ones(el.order + 1)}
        return op_dict

    def test_mesh_generator(self):
        tested_macro_grids_and_orders = [([0, 2],  [4]),
                                         ([0, 1, 2], [2, 2]),
                                         ([-0.5, 1.5, 2.3, 3], [4, 3, 2])]
        for (macro_grid, order) in tested_macro_grids_and_orders:
            yield self.check_residual_and_solution, Mesh1D(macro_grid, order, ['f', 'g'])


class TestLSProblemNonLinear(LSProblemNonLinear):
    """Class for testing a poisson problem in 1 variable on N elements."""

    def define_equations(self, el):
        # Solution: f(x) = 2-(x-1)^2
        x = el.x_1v
        f_old = self.f_old[self.mesh.gm[el.number][el.pos['f']]]
        op_dict = {'f.f': (f_old * el.dx.T).T,  # equivalent to and faster than "numpy.diag(f_old).dot(el.dx)"
                   'f': 2*x**3 - 6*x**2 + 2*x + 2}
        return op_dict

    def set_boundary_conditions(self):
        weight, left_value = 1.0, 1.0
        self.k_el[0][0, 0] += weight
        self.g_el[0][0] += weight * left_value

    def get_analytical_solution(self):
        x0 = self.mesh.elem[0].boundaries['x'][0]
        x1 = self.mesh.elem[-1].boundaries['x'][1]
        x = x0
        for el in self.mesh.elem:
            x = numpy.append(x, el.x_1v[1:])
        analytical_solution = 2 - (x - 1) ** 2  # assume f(0) = 1
        return analytical_solution

    def test_mesh_generator(self):
        tested_macro_grids_and_orders = [([0, 1, 2],  [4, 4]),
                                         ([0.0, 1, 1.5, 2.0], [4, 3, 3])]
        for (macro_grid, order) in tested_macro_grids_and_orders:
            yield self.check_residual_and_solution, Mesh1D(macro_grid, order, ['f'])

    def check_residual_and_solution(self, prob_mesh):
        self.__init__(prob_mesh)
        self.solve()
        assert_almost_equal(self.residual, 0.0)
        numpy.testing.assert_allclose(self.f[0:prob_mesh.dof_1v], *[self.get_analytical_solution()])


class TestLSProblemTimeSlab(LSProblemTimeSlab):

    def define_equations(self, el):
        id_mat = numpy.identity(el.order + 1)
        zero_vec = numpy.zeros(el.order + 1)
        dx = el.dx

        m, c, k = 1.0, 0.2, 1.0
        op_dict = {'v0.v0': m * dx + c * id_mat,
                   'v0.x0': k * id_mat,
                   'v0': zero_vec,

                   'x0.v0': id_mat,
                   'x0.x0': -dx,
                   'x0': zero_vec}
        return op_dict

    def set_boundary_conditions(self):
        weight = 1.0
        initial_speed = 0.0
        initial_position = 5.0
        x0_index = self.mesh.elem[0].pos['x0'][0]
        self.k_el[0][0, 0] += weight
        self.g_el[0][0] += weight * initial_speed
        self.k_el[0][x0_index, x0_index] += weight
        self.g_el[0][x0_index] += weight * initial_position

    def get_analytical_solution(self):
        m, c, k = 1.0, 0.2, 1.0
        x0 = self.mesh.elem[0].boundaries['x'][0]
        x = x0
        for el in self.mesh.elem:
            x = numpy.append(x, el.x_1v[1:])
        damp_coef = c / (2 * math.sqrt(m * k))
        nat_freq = math.sqrt(k / m)
        initial_position = 5.0
        # analytical_solution = initial_position * math.exp(-damp_coef * nat_freq * x) * math.cos(math.sqrt(1 - damp_coef ** 2) * nat_freq * x)
        analytical_solution = numpy.zeros(x.size)
        for idx, x_val in enumerate(x):
            analytical_solution[idx] = initial_position * math.exp(-damp_coef * nat_freq * x_val) * math.cos(math.sqrt(1 - damp_coef ** 2) * nat_freq * x_val)
        return analytical_solution

    def test_mesh_generator(self):
        tested_macro_grids_and_orders = [(numpy.linspace(0.0, 5.0, 20), [4] * 9)]
        for (macro_grid, order) in tested_macro_grids_and_orders:
            yield self.check_residual_and_solution, Mesh1D(macro_grid, order, ['v0', 'x0'])

    def check_residual_and_solution(self, prob_mesh):
        self.__init__(prob_mesh)
        self.solve()
        assert_almost_equal(self.residual, 0.0)

        # numpy.testing.assert_allclose(self.f[prob_mesh.dof_1v: 2*prob_mesh.dof_1v], *[self.get_analytical_solution()])
