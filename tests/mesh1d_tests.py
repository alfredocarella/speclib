from nose.tools import assert_equal, assert_almost_equal
import numpy
from solverls import speclib
from solverls.mesh1d import Mesh1d

__author__ = 'alfredoc'


def test_mesh1d():    # Testing the mesh generation (plotting is not tested here)
    macro_grid, orders = numpy.array((0.0, 1.0, 2.0, 3.0)), numpy.array((3, 4, 2))
    list_of_variables = ['T', 'pres', 'quality']

    my_mesh1d = Mesh1d(macro_grid, orders, list_of_variables)

    numpy.testing.assert_array_equal(my_mesh1d.element_orders, orders)
    numpy.testing.assert_allclose(my_mesh1d.macro_nodes, macro_grid)
    assert_equal(my_mesh1d.list_of_variables, list_of_variables)
    assert_equal(my_mesh1d.number_of_elements, len(my_mesh1d.element_orders))
    assert_equal(my_mesh1d.number_of_variables, len(my_mesh1d.list_of_variables))
    assert_equal(my_mesh1d.dof_1v, sum(my_mesh1d.element_orders)+1)
    assert_equal(my_mesh1d.dof_nv, my_mesh1d.number_of_variables*my_mesh1d.dof_1v)
    assert_equal(len(my_mesh1d.gm), my_mesh1d.number_of_elements)
    assert_equal(len(my_mesh1d.gm_1v), my_mesh1d.number_of_elements)

    integral_test_value = 0
    for el in range(my_mesh1d.number_of_elements):
        integral_test_value += my_mesh1d.quadrature_points[el].dot(my_mesh1d.quadrature_weights[el])
        assert_almost_equal(
            my_mesh1d.jac[el], (my_mesh1d.x[my_mesh1d.gm_1v[el][-1]]-my_mesh1d.x[my_mesh1d.gm_1v[el][0]])/2.0)
        numpy.testing.assert_array_equal(my_mesh1d.gm[el][:my_mesh1d.element_orders[el]+1], my_mesh1d.gm_1v[el])
        numpy.testing.assert_array_equal(my_mesh1d.quadrature_points[el], my_mesh1d.x[my_mesh1d.gm_1v[el]])
        numpy.testing.assert_array_equal(
            my_mesh1d.dx[el], speclib.lagrange_derivative_matrix_gll(my_mesh1d.element_orders[el]+1)/my_mesh1d.jac[el])
        numpy.testing.assert_array_equal(
            my_mesh1d.long_quadrature_weights[el],
            numpy.tile(my_mesh1d.quadrature_weights[el], my_mesh1d.number_of_variables))
        pos_var_test_value = []
        for var in my_mesh1d.list_of_variables:
            pos_var_test_value = numpy.append(pos_var_test_value, my_mesh1d.pos[el][var])
        numpy.testing.assert_array_equal(pos_var_test_value, range(len(my_mesh1d.gm[el])))
    assert_almost_equal(integral_test_value, (my_mesh1d.macro_nodes[-1]-my_mesh1d.macro_nodes[0])**2 / 2)