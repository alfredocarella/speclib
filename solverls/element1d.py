__author__ = 'alfredoc'
from solverls.speclib import lagrange_derivative_matrix_gll, gll
import numpy

class Element1D():
    """Spectral (high order) one-dimensional element. One of the elementary blocks that compose a mesh (and a mesh object)"""
    def __init__(self, element_boundaries, local_gm, variables):

        self.variables = variables
        self.order = (len(local_gm) // len(variables)) - 1
        self.pos = self.create_local_variable_indices(self.variables, self.order)

        self.boundaries = element_boundaries  # geometry
        self.quadrature_points, self.quadrature_weights = gll(self.order + 1, self.boundaries[0], self.boundaries[1])
        self.long_quadrature_weights = numpy.tile(self.quadrature_weights, len(variables))
        self.x = numpy.tile(self.quadrature_points, len(variables))

        self.jac = (self.boundaries[1] - self.boundaries[0]) / 2.0  # Derivatives
        self.dx = lagrange_derivative_matrix_gll(self.order + 1) / self.jac

    def create_local_variable_indices(self, variables, order):
        pos = {}
        for var_ in range(len(variables)):
            first_node_in_variable = var_ * (order + 1)
            last_node_in_variable = first_node_in_variable + (order + 1)
            pos[variables[var_]] = numpy.arange(first_node_in_variable, last_node_in_variable)
        return pos


if __name__ == '__main__':
    macro_nodes, gm_element, list_of_variables = [1, 2], numpy.arange(4, dtype=numpy.int), ['f1', 'f2']
    Element1D(macro_nodes, gm_element, list_of_variables)

    print('Test inputs:')
    print('macro_nodes = %s' % str(macro_nodes))
    print('gm_element = %s' % str(gm_element))
    print('list_of_variables = %s' % str(list_of_variables))

