from nose.tools import assert_equal, assert_almost_equal
import numpy
from solverls.mesh1d import Mesh1D
# from solverls.element1d import Element1D

__author__ = 'Alfredo Carella'


def test_mesh1d():    # Testing the mesh generation (plotting is not tested here)
    my_mesh1d = Mesh1D(macro_grid=[-1, 1, 2, 4], element_orders=[3, 4, 2], variable_names=['Var A', 'Var B', 'Var C'])
    # Testing mesh attributes
    numpy.testing.assert_allclose(my_mesh1d.macro_grid, [-1, 1, 2, 4])
    numpy.testing.assert_array_equal(my_mesh1d.element_orders, [3, 4, 2])
    assert_equal(my_mesh1d.variables, ['Var A', 'Var B', 'Var C'])
    assert_equal(my_mesh1d.dof, 30)
    # assert_equal(Element1D.get_num_instances(), 3)  # FIXME: static instance counter not working as intended
    # Testing list of elements
    numpy.testing.assert_array_equal(my_mesh1d.gm[0], [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23])
    numpy.testing.assert_array_equal(my_mesh1d.gm[1], [3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 23, 24, 25, 26, 27])
    numpy.testing.assert_array_equal(my_mesh1d.gm[2], [7, 8, 9, 17, 18, 19, 27, 28, 29])
    numpy.testing.assert_array_equal(my_mesh1d.elem[1].pos['Var B'], [5, 6, 7, 8, 9])

    domain_integral_a, domain_integral_b = 0, 0
    for el in my_mesh1d.elem:
        domain_integral_a += el.w_1v.dot(el.x_1v)
        domain_integral_b += el.w_1v.dot(el.x_1v ** 2)
    assert_almost_equal(domain_integral_a, (my_mesh1d.macro_grid[-1]**2 - my_mesh1d.macro_grid[0]**2) / 2)
    assert_almost_equal(domain_integral_b, (my_mesh1d.macro_grid[-1]**3 - my_mesh1d.macro_grid[0]**3) / 3)