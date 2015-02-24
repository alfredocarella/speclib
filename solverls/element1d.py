from solverls.speclib import lagrange_derivative_matrix_gll, gll
import numpy

__author__ = 'Alfredo Carella'


class Element1D():
    """Spectral (high order) 1D element. One of the elementary blocks that compose a mesh (and a mesh object)"""

    number_of_instances = 0

    def __init__(self, boundaries, order, variables):
        Element1D.number_of_instances += 1
        # Numbering
        self.number = Element1D.number_of_instances - 1
        self.variables = variables
        self.order = order
        self.pos = self.create_local_variable_indices()
        # Geometry
        self.boundaries = {'x': boundaries}
        self.x_1v, self.w_1v = gll(self.order + 1, self.boundaries['x'][0], self.boundaries['x'][1])
        self.w_nv = numpy.tile(self.w_1v, len(variables))
        self.x_nv = numpy.tile(self.x_1v, len(variables))
        # Differentiation
        self.jac = (self.boundaries['x'][1] - self.boundaries['x'][0]) / 2.0
        self.dx = lagrange_derivative_matrix_gll(self.order + 1) / self.jac

    def create_local_variable_indices(self):
        pos = {}
        dof_local = (self.order + 1)
        for var_ in range(len(self.variables)):
            pos[self.variables[var_]] = numpy.arange(var_ * dof_local, (var_ + 1) * dof_local)
        return pos

    def plot(self):
        raise NotImplementedError("Child classes must implement this method.")

    @staticmethod
    def get_num_instances():
        return Element1D.number_of_instances

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