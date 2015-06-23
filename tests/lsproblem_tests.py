import itertools

from nose.tools import assert_almost_equal
import numpy

from solverls.lsproblemlinear import LSProblem, LSProblemLinear
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
        LSProblem.__init__(self, prob_mesh)
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


class TestLSProblemNonLinearKK(LSProblemNonLinear):
    """Class for testing a poisson problem in 2 variables on N elements."""

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
        LSProblem.__init__(self, prob_mesh)
        self.solve()
        assert_almost_equal(self.residual, 0.0)
        numpy.testing.assert_allclose(self.f[0:prob_mesh.dof_1v], *[self.get_analytical_solution()])


# TODO: The following code still needs to be refactored
def test_problem_torsional_1v():
    """Testing a torsional vibration problem (1 mass)"""
    macro_grid = numpy.linspace(0.0, 30.0, 50)
    orders = [4] * (len(macro_grid)-1)
    list_of_variables = ['v0', 'x0']

    my_mesh1d = Mesh1D(macro_grid, orders, list_of_variables)
    my_problem = TorsionalProblemLinearTest(my_mesh1d)
    my_problem.solve()
    assert_almost_equal(my_problem.residual, 0.0, 6)

    # my_problem.plot()  # filename='testingProblemTorsional1v.pdf' #FIXME: Commented to avoid plotting during test execution

    print("The residual for this problem is %04.2e" % my_problem.residual)


def test_problem_torsional_nv():
    """Testing a torsional vibration problem (N masses)"""
    macro_grid = numpy.linspace(0.0, 30.0, 40)
    orders = [5] * (len(macro_grid)-1)
    number_of_masses = 2

    list_of_variables = []
    for variable_number in range(number_of_masses):
        list_of_variables.append('v%d' % variable_number)
        list_of_variables.append('x%d' % variable_number)
    print(list_of_variables)

    my_mesh1d = Mesh1D(macro_grid, orders, list_of_variables)
    my_problem = TorsionalProblemLinearTestNv(my_mesh1d)
    my_problem.solve()
    # assert_almost_equal(my_problem.residual, 0.0, 6)

    # my_problem.plot()  # filename='testingProblemTorsionalNv.pdf') #FIXME: Commented to avoid plotting during test execution

    print("The residual for this problem is %04.2e" % my_problem.residual)

    print("range(1,1) = %r" % range(1, 1))


# ********************************************************** #
# ********************** TESTING CODE ********************** #
# ********************************************************** #

class TorsionalProblemLinearTest(LSProblemTimeSlab):
    """Class for testing a torsional problem in N variables on N elements."""

    def define_equations(self, el):
        op_dict = {}

        id_mat = numpy.identity(el.order + 1)
        zero_vec = numpy.zeros(el.order + 1)
        dx = el.dx
        # f = numpy.diag(self.f[self.mesh.gm[el]]) # <--only for non-linear problems

        m = 1.0
        c = 0.2
        k = 1.0

        op_dict['v0.v0'] = m*dx + c*id_mat
        op_dict['v0.x0'] = k*id_mat
        op_dict['v0'] = zero_vec  # F

        op_dict['x0.v0'] = id_mat
        op_dict['x0.x0'] = -dx
        op_dict['x0'] = zero_vec  #

        return op_dict

    def set_boundary_conditions(self):
        weight = 1.0
        initial_speed = 0.0
        initial_position = 5.0
        self.k_el[0][0, 0] += weight
        self.g_el[0][0] += weight * initial_speed
        self.k_el[0][5, 5] += weight
        self.g_el[0][5] += weight * initial_position


class TorsionalProblemLinearTestNv(LSProblemTimeSlab):
    """Class for testing a torsional problem in N variables on N elements."""
    def define_equations(self, el):
        op_dict = {}
        x = el.x_1v
        id_mat = numpy.identity(el.order + 1)
        zero_vec = numpy.zeros(el.order + 1)
        dx_mat = el.dx
        # f = numpy.diag(self.f[el.nodes]) # <--only for non-linear problems

        m = [2.0, 4.0, 3.0, 10.0]
        c_abs = [1.0, 0.0, 0.0, 0.0]
        c = [0.0, 0.0, 0.0]
        k = [2.0, 7.0, 6.0]

        i = 0
        vi = 'v0'
        vip1 = 'v1'
        xi = 'x0'
        xip1 = 'x1'

        op_dict[vi + '.' + vi] = m[i]*dx_mat + (c[i]+c_abs[i])*id_mat
        op_dict[vi + '.' + xi] = k[i]*id_mat
        op_dict[vi + '.' + vip1] = -1.0*c[i]*id_mat
        op_dict[vi + '.' + xip1] = -1.0*k[i]*id_mat

        op_dict[xi + '.' + vi] = -1.0*id_mat
        op_dict[xi + '.' + xi] = dx_mat

        op_dict[vi] = numpy.sin(x/10.0)  # F_1

        n = len(el.variables) // 2 - 1
        for mass in range(1, n):
            vim1 = 'v'+str(n-1)
            vi = 'v'+str(n)
            vip1 = 'v'+str(n+1)

            xim1 = 'x'+str(n-1)
            xi = 'x'+str(n)
            xip1 = 'x'+str(n+1)

            op_dict[vi + '.' + vim1] = -1.0*c[i-1]*id_mat
            op_dict[vi + '.' + xim1] = -1.0*k[i-1]*id_mat
            op_dict[vi + '.' + vi] = m[i]*dx_mat + (c[i-1]+c[i]+c_abs[i])*id_mat
            op_dict[vi + '.' + xi] = (k[i-1]+k[i])*id_mat
            op_dict[vi + '.' + vip1] = -1.0*c[i]*id_mat
            op_dict[vi + '.' + xip1] = -1.0*k[i]*id_mat

            op_dict[xi + '.' + vi] = -1.0*id_mat
            op_dict[xi + '.' + xi] = dx_mat

        vim1 = 'v'+str(n-1)
        vi = 'v'+str(n)
        xim1 = 'x'+str(n-1)
        xi = 'x'+str(n)

        op_dict[vi + '.' + vim1] = -1.0*c[n-1]*id_mat
        op_dict[vi + '.' + xim1] = -1.0*k[n-1]*id_mat
        op_dict[vi + '.' + vi] = m[n]*dx_mat + (c[n-1]+c_abs[n])*id_mat
        op_dict[vi + '.' + xi] = k[n-1]*id_mat
        op_dict[xi + '.' + vi] = -1.0*id_mat
        op_dict[xi + '.' + xi] = dx_mat

        op_dict[vi] = zero_vec  # F_n

        return op_dict

    def set_boundary_conditions(self):
        weight = 10.0
        initial_value = {'v0': 0.0, 'x0': 0.0}
        for var in ['v0', 'x0']:
            var_index = self.mesh.elem[0].pos[var][0]
            self.k_el[0][var_index, var_index] += weight
            self.g_el[0][var_index] += weight * initial_value[var]


if __name__ == "__main__":
    my_test = TestLSProblemLinearNEl2Var()
    my_test.test_input_generator_Nel_2var()
