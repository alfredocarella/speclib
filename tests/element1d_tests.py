import itertools

from nose.tools import assert_equal
import numpy.testing

from solverls import spectral
from solverls.mesh1d import Element1D

__author__ = 'Alfredo Carella'


def element1d_test_case_generator():
    tested_orders = range(2, 7)
    tested_boundaries = [(-1, 1), (2, 5), (-6, -4), (-3, 1)]
    tested_varlists = [['f'], ['Temperature', 'Pressure', 'Quality'], []]
    for order, boundaries, var_list in itertools.product(*[tested_orders, tested_boundaries, tested_varlists]):
        yield check_consistency_in_element_generation, order, boundaries, var_list


def check_consistency_in_element_generation(order, boundaries, var_list):
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

