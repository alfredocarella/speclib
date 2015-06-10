from nose.tools import assert_equal
import numpy.testing

from solverls import spectral
from solverls.mesh1d import Element1D

__author__ = 'Alfredo Carella'


def test_element1d():
    boundaries = [2, 3]
    order = 3
    var_list = ['var_1', 'var_2', 'var_3']
    my_element = Element1D(boundaries, order, var_list)
    # Test numbering
    assert_equal(my_element.variables, var_list)
    assert_equal(my_element.order, order)
    for idx, var in enumerate(var_list):
        numpy.testing.assert_array_equal(my_element.pos[var], range(idx*(order+1), (idx+1)*(order+1)))
    # Test geometry
    assert_equal(my_element.boundaries['x'], boundaries)
    numpy.testing.assert_array_equal(my_element.x_1v, spectral.gll(order, boundaries[0], boundaries[1])[0])
    numpy.testing.assert_array_equal(my_element.w_1v, spectral.gll(order, boundaries[0], boundaries[1])[1])
    numpy.testing.assert_array_equal(my_element.x_nv, numpy.tile(my_element.x_1v, len(var_list)))
    numpy.testing.assert_array_equal(my_element.w_nv, numpy.tile(my_element.w_1v, len(var_list)))
    # Test differentiation
    assert_equal(my_element.jac, (boundaries[1] - boundaries[0]) / 2.0)
    numpy.testing.assert_array_equal(my_element.dx, spectral.gll_derivative_matrix(order) / my_element.jac)


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
