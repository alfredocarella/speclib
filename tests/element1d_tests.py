from nose.tools import assert_equal, assert_almost_equal
import numpy
from solverls import speclib
from solverls.mesh1d import Element1D

__author__ = 'Alfredo Carella'


# def test_element1d():    # Testing the mesh generation (plotting is not tested here)
#     macro_grid, orders = numpy.array((0.0, 1.0, 2.0, 3.0)), numpy.array((3, 4, 2))
#     list_of_variables = ['T', 'pres', 'quality']
#
#     my_mesh1d = Mesh1D(macro_grid, orders, list_of_variables)
#
#     numpy.testing.assert_array_equal(my_mesh1d.element_orders, orders)
#     numpy.testing.assert_allclose(my_mesh1d.macro_grid, macro_grid)
#     assert_equal(my_mesh1d.variables, list_of_variables)
#     assert_equal(len(my_mesh1d.elem), len(my_mesh1d.element_orders))
#     assert_equal(my_mesh1d.dof, len(my_mesh1d.variables) * (sum(my_mesh1d.element_orders)+1))
#     assert_equal(len(my_mesh1d.gm), len(my_mesh1d.elem))
#
#     integral_test_value = 0
#     for el in range(len(my_mesh1d.elem)):
#         integral_test_value += my_mesh1d.elem[el].x_1v.dot(my_mesh1d.elem[el].w_1v)
#         assert_almost_equal(
#             my_mesh1d.elem[el].jac, (my_mesh1d.elem[el].x_1v[-1]-my_mesh1d.elem[el].x_1v[0]) / 2.0)
#         numpy.testing.assert_array_equal(my_mesh1d.gm[el][:my_mesh1d.element_orders[el]+1], my_mesh1d.gm[el][0:(my_mesh1d.elem[el].order + 1)])
#         numpy.testing.assert_array_equal(my_mesh1d.elem[el].x_1v, my_mesh1d.elem[el].x_1v)
#         numpy.testing.assert_array_equal(
#             my_mesh1d.elem[el].dx, speclib.lagrange_derivative_matrix_gll(my_mesh1d.elem[el].order+1)/my_mesh1d.elem[el].jac)
#         numpy.testing.assert_array_equal(
#             my_mesh1d.elem[el].w_nv,
#             numpy.tile(my_mesh1d.elem[el].w_1v, len(my_mesh1d.variables)))
#         pos_var_test_value = []
#         for var in my_mesh1d.variables:
#             pos_var_test_value = numpy.append(pos_var_test_value, my_mesh1d.elem[el].pos[var])
#         numpy.testing.assert_array_equal(pos_var_test_value, range(len(my_mesh1d.gm[el])))
#     assert_almost_equal(integral_test_value, (my_mesh1d.macro_grid[-1]-my_mesh1d.macro_grid[0])**2 / 2)