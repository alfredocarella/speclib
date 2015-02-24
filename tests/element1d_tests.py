from nose.tools import assert_equal, assert_almost_equal
import numpy
from solverls import speclib
from solverls.mesh1d import Element1D

__author__ = 'Alfredo Carella'


def test_element1d():
    my_element = Element1D(boundaries=[2, 3], order=3, variables=['var_1', 'var_2'])
    assert_equal(Element1D.get_num_instances(), 1)
    # Test numbering
    assert_equal(my_element.variables, ['var_1', 'var_2'])
    assert_equal(my_element.order, 3)
    numpy.testing.assert_array_equal(my_element.pos['var_2'], [4, 5, 6, 7])
    # Test geometry
    assert_equal(my_element.boundaries['x'], [2.0, 3.0])
    numpy.testing.assert_array_equal(my_element.x_1v, speclib.gll(4, 2, 3)[0])
    numpy.testing.assert_array_equal(my_element.w_1v, speclib.gll(4, 2, 3)[1])
    numpy.testing.assert_array_equal(my_element.x_nv, numpy.tile(speclib.gll(4, 2, 3)[0], 2))
    numpy.testing.assert_array_equal(my_element.w_nv, numpy.tile(speclib.gll(4, 2, 3)[1], 2))
    # Test differentiation
    assert_equal(my_element.jac, 1/2.0)
    numpy.testing.assert_array_equal(my_element.dx, speclib.lagrange_derivative_matrix_gll(4) * 2)
