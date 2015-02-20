from nose.tools import assert_almost_equal, assert_equal
import numpy
from solverls.lsproblem import LSProblem
from solverls.mesh1d import Mesh1d

__author__ = 'alfredoc'


# TODO: These tests are not good at all. Proper tests to be implemented once the mesh class is correct.
def test_problem_1el_1v():
    """Testing results for a simple problem (1 var, 1 elem)"""
    macro_grid, orders, list_of_variables = numpy.array((0.0, 2.0)), numpy.array(4), ['f']
    my_mesh1d = Mesh1d(macro_grid, orders, list_of_variables)

    my_problem = TestLSProblem1el1v(my_mesh1d)

    my_problem.residual = my_problem.compute_residual()
    assert_almost_equal(my_problem.residual, 0.0)
    # numpy.testing.assert_allclose(
        # my_problem.Ke, my_problem.opL[0].T.dot(numpy.diag(my_problem.mesh.longQuadWeights[0])).dot(my_problem.opL[
        # 0])) #<--Missing BCs

    print("my_problem.opL = %r" % my_problem.op_l)
    print("my_problem.opG = %r" % my_problem.op_g)
    print("my_problem.Ke = %r" % my_problem.k_el)
    print("my_problem.Ge = %r" % my_problem.g_el)

    print("The residual for this problem is %04.2e" % my_problem.residual)
    print('\nThe solution vector is %r\n' % my_problem.f)

    print("Execution complete!")


def test_problem_nel_nv():
    """Testing a problem w/ multiple variables and elements"""
    macro_grid, orders, list_of_variables = numpy.array((0.0, 1.0, 2.0)), numpy.array((3, 3)), ['f', 'g']
    my_mesh1d = Mesh1d(macro_grid, orders, list_of_variables)
    numpy.testing.assert_array_equal(my_mesh1d.macro_grid, macro_grid)
    numpy.testing.assert_array_equal(my_mesh1d.element_orders, orders)
    assert_equal(my_mesh1d.variables, list_of_variables)

    my_problem = TestLSProblemNelNv(my_mesh1d)
    my_problem.residual = my_problem.compute_residual()
    my_problem.plot_solution(['f', 'g'], 'testingProblemNelNv.pdf')

    # print("my_problem.opL = %r" % my_problem.opL)
    # print("my_problem.opG = %r" % my_problem.opG)
    # print("my_problem.mesh.dx = %r" % my_problem.mesh.dx)
    # print("my_problem.mesh.gm = %r" % my_problem.mesh.gm)

    # print('\nThe "elemGM" solution vector is %r\n' % (my_problem.f))
    print("The residual for this problem is %04.2e" % my_problem.residual)

    print("testingProblemNelNv(): Execution complete!")


class TestLSProblem1el1v(LSProblem):
    """Class for testing a simple problem in 1 variable on 1 element."""
    def __init__(self, mesh):
        LSProblem.__init__(self, mesh)
        self.solve_linear()

    def set_equations(self, el):
        operator_size = len(self.mesh.gm[el]) / len(self.mesh.variables)

        op_l = {'f.f': self.mesh.dx[el].dot(self.mesh.dx[el])}
        op_g = {'f': -1.0 * numpy.ones(operator_size)}

        return op_l, op_g

    def set_boundary_conditions(self):
        weight = 1.0
        left_value = 3.0
        right_value = 3.0
        self.k_el[0][0, 0] += weight
        self.g_el[0][0] += weight * left_value
        self.k_el[-1][-1, -1] += weight
        self.g_el[-1][-1] += weight * right_value


class TestLSProblemNelNv(LSProblem):
    """Class for testing a poisson problem in 2 variables on N elements."""
    def __init__(self, mesh):
        LSProblem.__init__(self, mesh)
        self.solve_linear()

    def set_equations(self, el):
        operator_size = len(self.mesh.gm[el]) / len(self.mesh.variables)
        op_l = {'f.f': self.mesh.dx[el],
                'f.g': -1.0 * numpy.identity(operator_size),
                'g.f': numpy.zeros((operator_size, operator_size)),
                'g.g': self.mesh.dx[el]}

        op_g = {'f': numpy.zeros(operator_size),
                'g': -1.0 * numpy.ones(operator_size)}

        return op_l, op_g

    def set_boundary_conditions(self):
        weight = 1.0
        left_value = 3.0
        right_value = -1.0

        self.k_el[0][0, 0] += weight
        self.g_el[0][0] += weight * left_value
        self.k_el[-1][-1, -1] += weight
        self.g_el[-1][-1] += weight * right_value


# def testing_problem_non_linear():
#     """Testing iterative routine for solving a non-linear problem"""
#     macro_grid, orders, list_of_variables = numpy.array((0.0, 1.0, 2.0)), numpy.array((3, 3)), ['f']
#     print("macro_grid = %r - orders = %r - list_of_variables = %r" % (macro_grid, orders, list_of_variables))
#
#     my_mesh1d = Mesh1d(macro_grid, orders, list_of_variables)
#     my_problem = TestLSProblemNonLinear(my_mesh1d)
#     my_problem.plot_solution(['f'], 'testingProblemNonLinear.pdf')
#
#     print("The residual for this problem is %04.2e" % my_problem.residual)
#
#     print("Execution complete!")
#
#
# def test_problem_torsional_1v():
#     """Testing a torsional vibration problem (1 mass)"""
#     macro_grid = numpy.linspace(0.0, 30.0, 50)
#     orders = [4] * (len(macro_grid)-1)
#     list_of_variables = ['v0', 'x0']
#     # print("macro_grid = %r - orders = %r - list_of_variables = %r" % (macro_grid, orders, list_of_variables))
#
#     my_mesh1d = Mesh1d(macro_grid, orders, list_of_variables)
#     my_problem = TorsionalProblemTest(my_mesh1d)
#     my_problem.plot_solution()  # filename='testingProblemTorsional1v.pdf')
#
#     print("'TorsionalProblemTest.computeResidual()' does not work.")
#
#     # The following line will not work because my_problem.opL and and my_problem.opG have been reduced to 1 element
#     # and therefore the full problem information is not saved
#     # print("The residual for this problem is %04.2e" % my_problem.computeResidual())
#
#     my_memo = """
#     2013-12-04: A MINIMUM EXAMPLE IS WORKING!!! :-)
#
#     Check-list for project (pending tasks):
#     -  DONE! Add support for multi-equation
#     -  DONE! Add support for multi-element
#     -  DONE! Solve element by element
#     -  DONE! Plot problem solutions (good enough now, can be improved later)
#     -  DONE! Implement a solver routine for non-linear systems (Picard iteration is working)
#     -  DONE! Implement a time-slab approach for only one time element at the time
#     -  Try first mechanical problem (N variables with a self-generated system of equations)
#     -  Consider creating animations (rotating masses / springs) before presenting script
#     -  Consider including a steady-state Fourier analysis module
#     - self.computeResidual(self) <--Compliance with BC not considered yet!
#     - Find out how to do a code profiling
#     """
#
#     print(my_memo + '\n' + "testingProblemTorsional1v(): Execution complete!")
#
#
# def test_problem_torsional_nv():
#     """Testing a torsional vibration problem (N masses)"""
#     macro_grid = numpy.linspace(0.0, 30.0, 40)
#     orders = [5] * (len(macro_grid)-1)
#     number_of_masses = 2
#
#     list_of_variables = []
#     for variable_number in range(number_of_masses):
#         list_of_variables.append('v%d' % variable_number)
#         list_of_variables.append('x%d' % variable_number)
#     print(list_of_variables)
#
#     my_mesh1d = Mesh1d(macro_grid, orders, list_of_variables)
#     my_problem = TorsionalProblemTestNv(my_mesh1d)
#     my_problem.solve_linear_slab()
#     my_problem.plot_solution()  # filename='testingProblemTorsionalNv.pdf')
#
#     print("'TorsionalProblemTestNv.computeResidual()' does not work.")
#     # The following line will not work because my_problem.opL and and my_problem.opG have been reduced to 1 element
#     # and therefore the full problem information is not saved
#     # print("The residual for this problem is %04.2e" % my_problem.computeResidual())
#
#     my_memo = """
#     2013-12-??: A MINIMUM EXAMPLE IS WORKING!!! :-)
#
#     Check-list for project (pending tasks):
#     -  Implement a non-linear time-slab approach
#     -  Try first mechanical problem (N variables with a self-generated system of equations)
#     -  Consider creating animations (rotating masses / springs) before presenting script
#     -  Consider including a steady-state Fourier analysis module
#     - self.computeResidual(self) <--Compliance with BC not considered yet!
#     - Find out how to do a code profiling
#     """
#
#     print(my_memo + '\n' + "testingProblemTorsionalNv(): Execution complete!")
#
#     print("range(1,1) = %r" % range(1, 1))


# ********************************************************** #
# ********************** TESTING CODE ********************** #
# ********************************************************** #




# class TestLSProblemNonLinear(LSProblem):
#     """Class for testing a poisson problem in 2 variables on N elements."""
#     def __init__(self, mesh):
#         LSProblem.__init__(self, mesh)
#         self.solve_nonlinear()
#
#     def set_equations(self, el):
#         # operator_size = len(self.mesh.gm[el]) / self.mesh.number_of_variables
#         op_l = {}
#         op_g = {}
#         x = self.mesh.x[self.mesh.gm[el]]
#
#         op_l['f.f'] = numpy.diag(self.f[self.mesh.gm[el]]).dot(self.mesh.dx[el])
#
#         op_g['f'] = 2*x**3 - 6*x**2 + 2*x + 2
#
#         return op_l, op_g
#
#     def set_boundary_conditions(self):
#         weight = 1.0
#         left_value = 1.0
#         self.k_el[0][0, 0] += weight
#         self.g_el[0][0] += weight * left_value
#
#
# class TorsionalProblemTest(LSProblem):
#     """Class for testing a torsional problem in N variables on N elements."""
#     def __init__(self, mesh):
#         LSProblem.__init__(self, mesh)
#         self.solve_linear_slab()
#
#     def set_equations(self, el):
#         op_l = {}
#         op_g = {}
#         operator_size = len(self.mesh.gm[el]) / self.mesh.number_of_variables
#
#         id_mat = numpy.identity(operator_size)
#         zero_vec = numpy.zeros(operator_size)
#         dx = self.mesh.dx[el]
#         # f = numpy.diag(self.f[self.mesh.gm[el]]) # <--only for non-linear problems
#
#         m = 1.0
#         c = 0.2
#         k = 1.0
#
#         op_l['v0.v0'] = m*dx + c*id_mat
#         op_l['v0.x0'] = k*id_mat
#         op_l['x0.v0'] = id_mat
#         op_l['x0.x0'] = -dx
#
#         op_g['v0'] = zero_vec  # F
#         op_g['x0'] = zero_vec  #
#
#         return op_l, op_g
#
#     def set_boundary_conditions(self):
#         weight = 1.0
#         initial_speed = 0.0
#         initial_position = 5.0
#         self.k_el[0][0, 0] += weight
#         self.g_el[0][0] += weight * initial_speed
#         self.k_el[0][5, 5] += weight
#         self.g_el[0][5] += weight * initial_position
#
#
# class TorsionalProblemTestNv(LSProblem):
#     """Class for testing a torsional problem in N variables on N elements."""
#     def set_equations(self, el):
#         op_l = {}
#         op_g = {}
#         operator_size = len(self.mesh.gm[el]) / self.mesh.number_of_variables
#         x = self.mesh.x[self.mesh.gm[el][:operator_size]]
#         id_mat = numpy.identity(operator_size)
#         zero_vec = numpy.zeros(operator_size)
#         dx_mat = self.mesh.dx[el]
#         # f = numpy.diag(self.f[self.mesh.gm[el]]) # <--only for non-linear problems
#
#         m = [2.0, 4.0, 3.0, 10.0]
#         c_abs = [1.0, 0.0, 0.0, 0.0]
#         c = [0.0, 0.0, 0.0]
#         k = [2.0, 7.0, 6.0]
#
#         i = 0
#         vi = 'v0'
#         vip1 = 'v1'
#         xi = 'x0'
#         xip1 = 'x1'
#
#         op_l[vi + '.' + vi] = m[i]*dx_mat + (c[i]+c_abs[i])*id_mat
#         op_l[vi + '.' + xi] = k[i]*id_mat
#         op_l[vi + '.' + vip1] = -1.0*c[i]*id_mat
#         op_l[vi + '.' + xip1] = -1.0*k[i]*id_mat
#
#         op_l[xi + '.' + vi] = -1.0*id_mat
#         op_l[xi + '.' + xi] = dx_mat
#
#         op_g[vi] = numpy.sin(x/10.0)  # F_1
#
#         n = int(self.mesh.number_of_variables/2 - 1)
#         for mass in range(1, n):
#             vim1 = 'v'+str(n-1)
#             vi = 'v'+str(n)
#             vip1 = 'v'+str(n+1)
#
#             xim1 = 'x'+str(n-1)
#             xi = 'x'+str(n)
#             xip1 = 'x'+str(n+1)
#
#             op_l[vi + '.' + vim1] = -1.0*c[i-1]*id_mat
#             op_l[vi + '.' + xim1] = -1.0*k[i-1]*id_mat
#             op_l[vi + '.' + vi] = m[i]*dx_mat + (c[i-1]+c[i]+c_abs[i])*id_mat
#             op_l[vi + '.' + xi] = (k[i-1]+k[i])*id_mat
#             op_l[vi + '.' + vip1] = -1.0*c[i]*id_mat
#             op_l[vi + '.' + xip1] = -1.0*k[i]*id_mat
#
#             op_l[xi + '.' + vi] = -1.0*id_mat
#             op_l[xi + '.' + xi] = dx_mat
#
#         vim1 = 'v'+str(n-1)
#         vi = 'v'+str(n)
#         xim1 = 'x'+str(n-1)
#         xi = 'x'+str(n)
#
#         op_l[vi + '.' + vim1] = -1.0*c[n-1]*id_mat
#         op_l[vi + '.' + xim1] = -1.0*k[n-1]*id_mat
#         op_l[vi + '.' + vi] = m[n]*dx_mat + (c[n-1]+c_abs[n])*id_mat
#         op_l[vi + '.' + xi] = k[n-1]*id_mat
#         op_l[xi + '.' + vi] = -1.0*id_mat
#         op_l[xi + '.' + xi] = dx_mat
#
#         op_g[vi] = zero_vec  # F_n
#
#         return op_l, op_g
#
#     def set_boundary_conditions(self):
#         initial_speed = 0.0
#         initial_position = 0.0
#
#         weight = 10.0
#         x0index = self.mesh.element_orders[0] + 1
#
#         self.k_el[0][0, 0] += weight
#         self.g_el[0][0] += weight * initial_speed
#         self.k_el[0][x0index, x0index] += weight
#         self.g_el[0][x0index] += weight * initial_position