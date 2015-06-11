import numpy

from solverls.spectral import gll_derivative_matrix, gll


__author__ = 'Alfredo Carella'


class Element1D:
    """Spectral 1D element. One of the elementary blocks that compose a mesh"""

    def __init__(self, boundaries, order, variables, number=-1):
        # Numbering
        self.number = number
        self.order = order
        self.variables = variables
        self.pos = create_variable_indices(order, variables)
        # Geometry
        self.boundaries = {'x': boundaries}
        self.x_1v, self.w_1v = gll(self.order, boundaries[0], boundaries[1])
        self.w_nv = numpy.tile(self.w_1v, len(variables))
        self.x_nv = numpy.tile(self.x_1v, len(variables))
        # Differentiation
        self.jac = (boundaries[1] - boundaries[0]) / 2.0
        self.dx = gll_derivative_matrix(self.order) / self.jac

    # def plot(self):
    #     raise NotImplementedError("Child classes must implement this method.")


def create_variable_indices(order, variables):
    pos = {}
    dof_local = (order + 1)
    for idx, var in enumerate(variables):
        pos[var] = numpy.arange(idx * dof_local, (idx + 1) * dof_local)
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
        print('self.number = %s' % my_element1d.number)
        print('self.variables = %s' % my_element1d.variables)
        print('self.order = %s' % my_element1d.order)
        print('self.pos = %s' % my_element1d.pos)
        print('# Geometric properties')
        print('self.boundaries = %s' % my_element1d.boundaries)
        print('self.quadrature_points = %s' % my_element1d.x_1v)
        print('self.quadrature_weights = %s' % my_element1d.w_1v)
        print('self.long_quadrature_weights = %s' % my_element1d.w_nv)
        print('self.x = %s' % my_element1d.x_nv)
        print('#Differentiation')
        print('self.jac = %s' % my_element1d.jac)
        print('self.dx = \n%s' % my_element1d.dx)

    minimum_working_example()