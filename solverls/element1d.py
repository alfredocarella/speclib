__author__ = 'alfredoc'
from solverls.speclib import lagrange_derivative_matrix_gll, gll
import numpy


class Element1D():
    """Spectral (high order) 1D element. One of the elementary blocks that compose a mesh (and a mesh object)"""

    def __init__(self, boundaries, order, variables):
        # Numbering
        self.variables = variables
        self.order = order
        self.pos = self.create_local_variable_indices()
        # Geometry
        self.boundaries = boundaries
        self.quadrature_points, self.quadrature_weights = gll(self.order + 1, self.boundaries[0], self.boundaries[1])
        self.long_quadrature_weights = numpy.tile(self.quadrature_weights, len(variables))
        self.x = numpy.tile(self.quadrature_points, len(variables))
        # Differentiation
        self.jac = (self.boundaries[1] - self.boundaries[0]) / 2.0
        self.dx = lagrange_derivative_matrix_gll(self.order + 1) / self.jac

    def create_local_variable_indices(self):
        pos = {}
        for var_ in range(len(self.variables)):
            first_node_in_variable = var_ * (self.order + 1)
            last_node_in_variable = first_node_in_variable + (self.order + 1)
            pos[self.variables[var_]] = numpy.arange(first_node_in_variable, last_node_in_variable)
        return pos


if __name__ == '__main__':
    def minimum_working_example():
        element_boundaries, element_order, list_of_variables = [1, 2], 3, ['f1', 'f2']
        my_element1d = Element1D(element_boundaries, element_order, list_of_variables)

        print('Test inputs:')
        print('element_boundaries = %s' % str(element_boundaries))
        print('element_order = %s' % str(element_order))
        print('list_of_variables = %s' % str(list_of_variables))
        print('')
        print('Internal variables:')
        print('# Numbering')
        print('self.variables = %s' % my_element1d.variables)
        print('self.order = %s' % my_element1d.order)
        print('self.pos = %s' % my_element1d.pos)
        print('# Geometric properties')
        print('self.boundaries = %s' % my_element1d.boundaries)
        print('self.quadrature_points = %s' % my_element1d.quadrature_points)
        print('self.quadrature_weights = %s' % my_element1d.quadrature_weights)
        print('self.long_quadrature_weights = %s' % my_element1d.long_quadrature_weights)
        print('self.x = %s' % my_element1d.x)
        print('#Differentiation')
        print('self.jac = %s' % my_element1d.jac)
        print('self.dx = \n%s' % my_element1d.dx)

    minimum_working_example()