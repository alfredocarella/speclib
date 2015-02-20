__author__ = 'alfredoc'
from solverls.speclib import lagrange_derivative_matrix_gll, gll
import numpy

class Element1D():
    """Spectral (high order) one-dimensional element. One of the elementary blocks that compose a mesh (and a mesh object)"""
    def __init__(self, element_boundaries, local_gm, variables):
        # Numbering
        self.variables = variables
        self.order = (len(local_gm) // len(variables)) - 1
        self.pos = self.create_local_variable_indices()
        # Geometric properties
        self.boundaries = element_boundaries
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
    macro_nodes, gm_element, list_of_variables = [1, 2], numpy.arange(8, dtype=numpy.int), ['f1', 'f2']
    my_element1d = Element1D(macro_nodes, gm_element, list_of_variables)

    print('Test inputs:')
    print('macro_nodes = %s' % str(macro_nodes))
    print('gm_element = %s' % str(gm_element))
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
