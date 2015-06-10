from nose.tools import assert_equal
import numpy.testing

from solverls import spectral
from solverls.mesh1d import Element1D

__author__ = 'Alfredo Carella'


def test_element1d():
    varlist=['var_1', 'var_2']
    my_element = Element1D(boundaries=[2, 3], order=3, variables=varlist)
    # assert_equal(Element1D.get_num_instances(), 1)
    # Test numbering
    assert_equal(my_element.variables, varlist)
    assert_equal(my_element.order, 3)
    numpy.testing.assert_array_equal(my_element.pos['var_2'], [4, 5, 6, 7])
    # Test geometry
    assert_equal(my_element.boundaries['x'], [2.0, 3.0])
    numpy.testing.assert_array_equal(my_element.x_1v, spectral.gll(4, 2, 3)[0])
    numpy.testing.assert_array_equal(my_element.w_1v, spectral.gll(4, 2, 3)[1])
    numpy.testing.assert_array_equal(my_element.x_nv, numpy.tile(spectral.gll(4, 2, 3)[0], 2))
    numpy.testing.assert_array_equal(my_element.w_nv, numpy.tile(spectral.gll(4, 2, 3)[1], 2))
    # Test differentiation
    assert_equal(my_element.jac, 1/2.0)
    numpy.testing.assert_array_equal(my_element.dx, spectral.lagrange_derivative_matrix_gll(4) * 2)


# class TestElement1d:
#
#     def setUp(self):
#         self.list_of_orders = list(range(2, 7))
#         self.list_of_boundary_pairs = [(-1, 1), (2, 5), (-6, -4), (-3, 1)]
#         self.list_of_variable_sets = ['f', ('var_1', 'var_2'), ('Temperature', 'Pressure', 'Quality')]
#
#     def tearDown(self):
#         del self.list_of_orders
#         del self.list_of_boundary_pairs
#         del self.list_of_variable_sets
#
#     def generate_set_of_gll_points_and_weights(self):
#         for order in self.list_of_orders:
#             for boundary_tuple in self.list_of_boundary_pairs:
#                 for varlist in self.list_of_variable_sets:
#                     yield Element1D(boundary_tuple, order, varlist)
#
#     def test_gll(self):
#         for points, weights in self.generate_set_of_gll_points_and_weights():
#             nose.tools.assert_almost_equal(numpy.sum(weights), points[-1] - points[0], msg="int(1)^{x_max}_{x_min} == x_max - x_min")
#             nose.tools.assert_almost_equal(weights.dot(points), (points[-1]**2 - points[0]**2)/2.0, msg="int(x)^{x_max}_{x_min} == (x_max^2 - x_min^2)/2")
