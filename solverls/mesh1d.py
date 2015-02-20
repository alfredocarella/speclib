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
        # Mesh attributes
        self.macro_grid = macro_grid
        self.element_orders = numpy.atleast_1d(element_orders)
        self.variables = variable_names
        self.dof = (numpy.sum(self.element_orders) + 1) * len(self.variables)

        # List of elements
        self.elem = []
        for el in range(len(self.macro_grid) - 1):
            self.elem.append(Element1D(self.macro_grid[el:el+2], self.element_orders[el], self.variables))

        # Gathering matrix (1:CREATE GM FROM ELEMENTS - 2:EVALUATE DELETING Mesh1d.x)
        self.gm = self.create_gm()  # mesh attribute
        self.x = numpy.zeros(self.dof)  # mesh attribute
        for el_ in range(len(self.elem)):
            self.x[self.gm[el_]] = numpy.tile(self.elem[el_].quadrature_points, len(self.variables))

    def create_gm(self):
        gm = []
        node_counter = 0
        for el_ in range(len(self.element_orders)):
            element_size = len(self.variables) * (self.element_orders[el_] + 1)
            gm.append(numpy.zeros(element_size, dtype=numpy.int))
            for var_ in range(len(self.variables)):
                start_position = var_ * (self.element_orders[el_] + 1)
                end_position = start_position + self.element_orders[el_] + 1
                start_number = var_ * self.dof // len(self.variables) + node_counter
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
        for node_ in range(self.dof // len(self.variables)):
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
        mesh_text = """
        Degrees of freedom per variable: {1}
        Total degrees of freedom: {1}
        Number of elements: {2}
        Variables: {3} {4}
        """.format(self.dof // len(self.variables), self.dof, len(self.elem), len(self.variables), self.variables)
        matplotlib.pyplot.text((macro_grid[0] + macro_grid[-1]) / 4.0, -0.9, mesh_text)
        matplotlib.pyplot.title('1-D mesh information')
        matplotlib.pyplot.xlabel('Independent variable coordinate')
        matplotlib.pyplot.axis([macro_grid[0], macro_grid[-1], -1, 1])
        matplotlib.pyplot.show()


if __name__ == '__main__':
    macro_coordinates, orders = numpy.array((0.0, 1.0, 2.0, 3.0)), numpy.array((3, 4, 2))
    list_of_variables = ['Temperature', 'Pressure', 'Quality']
    my_mesh1d = Mesh1d(macro_coordinates, orders, list_of_variables)

    print('Test inputs:')
    print('macro_grid = %s' % my_mesh1d.macro_grid)
    print('orders = %s' % my_mesh1d.element_orders)
    print('list_of_variables = %s' % my_mesh1d.variables)
    print('')
    print('Test outputs:')
    print("my_mesh1d.elem[0].dx = \n%r" % my_mesh1d.elem[0].dx)
    print("my_mesh1d.elem[0].quadrature_points = %r" % my_mesh1d.elem[0].quadrature_points)
    print("my_mesh1d.elem[0].dx.dot(my_mesh1d.elem[0].quadPoints) = \n%r" % my_mesh1d.elem[0].dx.dot(my_mesh1d.elem[0].quadrature_points))
    # my_mesh1d.plot_mesh()

