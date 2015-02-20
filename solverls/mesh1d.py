import matplotlib.pyplot
import numpy
from solverls.speclib import lagrange_derivative_matrix_gll, gll
from solverls.element1d import Element1D

__author__ = 'Alfredo Carella'


class Mesh1d(object):
    """
    This class represents a 1-dimensional high-order Finite Element mesh.
    macroGrid = vector with the position of element boundary nodes
    P = vector with the order for each element (int)
    NV = number of variables (int)
    """

    def __init__(self, macro_grid, element_orders, variable_names=('f',)):
        self.macro_grid = macro_grid  # mesh attribute
        self.element_orders = numpy.atleast_1d(element_orders)  # mesh attribute
        self.list_of_variables = variable_names  # mesh attribute

        self.number_of_elements = len(self.macro_grid) - 1  # mesh attribute (redundant)
        self.number_of_variables = len(self.list_of_variables)  # mesh attribute (redundant)

        self.quadrature_weights = []  # element attribute
        self.quadrature_points = []  # element attribute
        self.jac = []  # element attribute
        self.dx = []  # element attribute
        self.long_quadrature_weights = []  # element attribute
        self.pos = []  # element attribute
        for el_ in range(self.number_of_elements):
            lower_element_boundary = self.macro_grid[el_]
            upper_element_boundary = self.macro_grid[el_ + 1]
            self.jac.append((upper_element_boundary - lower_element_boundary) / 2.0)
            self.dx.append(lagrange_derivative_matrix_gll(self.element_orders[el_] + 1) / self.jac[el_])
            qx, qw = gll(self.element_orders[el_] + 1, lower_element_boundary, upper_element_boundary)
            self.quadrature_points.append(qx)  # x coordinate (quad points)
            self.quadrature_weights.append(qw)  # quadrature weights
            long_qw = numpy.tile(qw, self.number_of_variables)
            self.long_quadrature_weights.append(long_qw)
            self.pos.append({})
            for var_ in range(self.number_of_variables):
                first_node_in_element = var_ * (self.element_orders[el_] + 1)
                last_node_in_element = first_node_in_element + self.element_orders[el_] + 1
                self.pos[el_][self.list_of_variables[var_]] = numpy.arange(first_node_in_element, last_node_in_element)

        self.dof = (numpy.sum(self.element_orders) + 1) * self.number_of_variables  # mesh attribute
        self.gm = self.create_gm()  # mesh attribute

        self.x = numpy.zeros(self.dof)  # mesh attribute
        for el_ in range(self.number_of_elements):
            self.x[self.gm[el_]] = numpy.tile(self.quadrature_points[el_], self.number_of_variables)

    def create_gm(self):
        gm = []
        node_counter = 0
        for el_ in range(len(self.element_orders)):
            element_size = self.number_of_variables * (self.element_orders[el_] + 1)
            gm.append(numpy.zeros(element_size, dtype=numpy.int))
            for var_ in range(self.number_of_variables):
                start_position = var_ * (self.element_orders[el_] + 1)
                end_position = start_position + self.element_orders[el_] + 1
                start_number = var_ * self.dof // self.number_of_variables + node_counter
                end_number = start_number + self.element_orders[el_] + 1
                gm[el_][start_position: end_position] = numpy.arange(start_number, end_number)
            node_counter += self.element_orders[el_]
        return gm

    def plot_mesh(self):
        # Plot nodes and line
        micro_grid = self.x
        macro_grid = self.macro_grid
        matplotlib.pyplot.plot((macro_grid[0], macro_grid[-1]), (0, 0), 'r--', linewidth=2.0)  # Lines
        matplotlib.pyplot.plot(micro_grid, micro_grid * 0, 'ro')  # Nodes (micro)
        matplotlib.pyplot.plot(macro_grid, macro_grid * 0, 'bs', markersize=10)  # Nodes (macro)

        # Plot node and element numbers
        for node_ in range(self.dof // self.number_of_variables):
            matplotlib.pyplot.text(self.x[node_], -0.1, str(node_), fontsize=10, color='red')
        for border_ in range(len(macro_grid) - 1):
            element_center = (macro_grid[border_] + macro_grid[border_ + 1]) / 2.0
            matplotlib.pyplot.text(element_center, +0.1, str(border_), fontsize=15, color='blue')

        # Write annotations
        first_element_center = (macro_grid[0] + macro_grid[1]) / 2.0
        matplotlib.pyplot.annotate('element numbers', xy=(first_element_center, 0.17),
                                   xytext=(first_element_center, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
        matplotlib.pyplot.annotate('node number', xy=(micro_grid[1], -0.12), xytext=(micro_grid[1], -0.3),
                                   arrowprops=dict(facecolor='black', shrink=0.05))
        mesh_text = '''
        Degrees of freedom per variable: {0:d}
        Total degrees of freedom: {1:d}
        Number of elements: {2:d}
        Variables: {3:d} {4:s}
        '''.format(self.dof // self.number_of_variables, self.dof, self.number_of_elements, self.number_of_variables, self.list_of_variables)
        matplotlib.pyplot.text((macro_grid[0] + macro_grid[-1]) / 4.0, -0.9, mesh_text)
        matplotlib.pyplot.title('1-D mesh information')
        matplotlib.pyplot.xlabel('Independent variable coordinate')
        matplotlib.pyplot.axis([macro_grid[0], macro_grid[-1], -1, 1])
        matplotlib.pyplot.show()


if __name__ == '__main__':
    macro_coordinates, orders = numpy.array((0.0, 1.0, 2.0, 3.0)), numpy.array((3, 4, 2))
    list_of_variables = ['T', 'pres', 'quality']
    my_mesh1d = Mesh1d(macro_coordinates, orders, list_of_variables)

    print('Test inputs:')
    print('macro_grid = %s' % my_mesh1d.macro_grid)
    print('orders = %s' % my_mesh1d.element_orders)
    print('list_of_variables = %s' % my_mesh1d.list_of_variables)
    print('')

    # Example:
    # macroGrid, P, NV = numpy.array((0.0,1.0,2.0,3.0)), numpy.array((3,4,2)), 2
    # myMesh1d = Mesh1d(macroGrid, P, NV)
    # print("myMesh1d.Dx[0] = \n%r" % myMesh1d.Dx[0])
    # print("myMesh1d.x[myMesh1d.gm1v[0]] = %r" % myMesh1d.x[myMesh1d.gm_1v[0]])
    # print("myMesh1d.Dx[0].dot(myMesh1d.quadPoints[0]) = \n%r\" % myMesh1d.Dx[0].dot(myMesh1d.quadPoints[0]))
