import itertools

from nose.tools import assert_equal, assert_almost_equal
import numpy

from solverls.mesh1d import Mesh1D

__author__ = 'Alfredo Carella'


def mesh1d_test_case_generator():
    tested_macro_grids_and_orders = [([0, 1, 2, 3],  [2, 2, 2]),
                                     ([-1, 1, 2, 4], [3, 4, 2])]
    tested_varlists = [['f'], ['Var A', 'Var B', 'Var C'], []]
    for (macro_grid, element_orders), var_list in itertools.product(*[tested_macro_grids_and_orders, tested_varlists]):
        yield check_consistency_in_mesh1d, macro_grid, element_orders, var_list


# Testing the mesh generation (plotting is not tested here)
def check_consistency_in_mesh1d(macro_grid, element_orders, var_list):
    my_mesh1d = Mesh1D(macro_grid, element_orders, var_list)

    # Testing mesh attributes
    numpy.testing.assert_allclose(my_mesh1d.macro_grid, macro_grid)
    numpy.testing.assert_array_equal(my_mesh1d.element_orders, element_orders)
    assert_equal(my_mesh1d.variables, var_list)
    assert_equal(my_mesh1d.dof, (sum(element_orders) + 1) * len(var_list))

    # Testing list of elements
    for idx_var, var in enumerate(var_list):
        for idx_el, element in enumerate(my_mesh1d.elem):
            if idx_var == 0:
                mesh_pos_1 = my_mesh1d.gm[idx_el][element.pos[var]][0]
                mesh_pos_2 = sum(my_mesh1d.elem[el].order for el in range(idx_el))
                numpy.testing.assert_array_equal(mesh_pos_1, mesh_pos_2)
            else:
                mesh_pos_1 = my_mesh1d.gm[idx_el][element.pos[previous_var]] + my_mesh1d.dof_1v
                mesh_pos_2 = my_mesh1d.gm[idx_el][element.pos[var]]
                numpy.testing.assert_array_equal(mesh_pos_1, mesh_pos_2)
        previous_var = var

    # Testing linear and quadratic integration
    domain_integral_a = sum(el.w_1v.dot(el.x_1v) for el in my_mesh1d.elem)
    domain_integral_b = sum(el.w_1v.dot(el.x_1v ** 2) for el in my_mesh1d.elem)
    assert_almost_equal(domain_integral_a, (my_mesh1d.macro_grid[-1]**2 - my_mesh1d.macro_grid[0]**2) / 2)
    assert_almost_equal(domain_integral_b, (my_mesh1d.macro_grid[-1]**3 - my_mesh1d.macro_grid[0]**3) / 3)
