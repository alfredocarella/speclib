import numpy

from solverls.iterator import Iterator
from solverls.lsproblem import LSProblem


__author__ = 'Alfredo Carella'


class LSProblemNonLinear(LSProblem):

    def solve(self):
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
    class TestLSProblemNonLinear(LSProblemNonLinear):
        """Class for testing a poisson problem in 2 variables on N elements."""

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
        my_problem.solve()
        my_problem.plot()

        print('Test inputs:')
        print('macro_grid = %s' % my_mesh1d.macro_grid)
        print('orders = %s' % my_mesh1d.element_orders)
        print('variables = %s' % my_mesh1d.variables)
        print('')
        print('Test outputs:')
        print("The residual for this problem is %04.2e" % my_problem.residual)
    minimum_nonlinear_example()