import matplotlib.pyplot
import numpy
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
        # Gathering matrix
        self.gm = self.create_gm()  # mesh attribute

    def create_gm(self):
        gm = []
        node_counter = 0
        for el in self.elem:
            dof_element = len(self.variables) * (el.order + 1)
            gm.append(numpy.zeros(dof_element, dtype=numpy.int))
            for var_counter in range(len(self.variables)):
                start_position = var_counter * (el.order + 1)
                end_position = start_position + el.order + 1
                start_number = var_counter * self.dof // len(self.variables) + node_counter
                end_number = start_number + el.order + 1
                gm[-1][start_position: end_position] = numpy.arange(start_number, end_number)
            node_counter += el.order
        return gm

    def plot(self):
        # Plot nodes and element boundaries
        for el in self.elem:
            matplotlib.pyplot.plot((el.boundaries[0], el.boundaries[-1]), (0, 0), 'r--', linewidth=1.0)
            matplotlib.pyplot.plot(el.x_1v, el.x_1v * 0, 'ro')
            matplotlib.pyplot.plot(el.boundaries, el.boundaries * 0, 'bs', markersize=10)
            matplotlib.pyplot.text(sum(el.boundaries) / 2.0, +0.1, str(el.number), fontsize=15, color='blue')
            for node in range(el.order+1):
                matplotlib.pyplot.text(el.x_1v[node], -0.1, str(self.gm[el.number][node]), fontsize=10, color='red')

        # Write annotations
        macro_grid = self.macro_grid
        mesh_text = """
        Degrees of freedom: {0}
        Number of elements: {1}
        Variables: {2} {3}
        """.format(self.dof, len(self.elem), len(self.variables), self.variables)
        matplotlib.pyplot.text((macro_grid[0] + macro_grid[-1]) / 4.0, -0.9, mesh_text)
        matplotlib.pyplot.title('1-D mesh information')
        matplotlib.pyplot.xlabel('Independent variable coordinate')
        matplotlib.pyplot.axis([macro_grid[0], macro_grid[-1], -1, 1])
        matplotlib.pyplot.show()


if __name__ == '__main__':
    def minimum_working_example():
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
        print("my_mesh1d.elem[0].x_1v = %r" % my_mesh1d.elem[0].x_1v)
        print("my_mesh1d.elem[0].dx.dot(my_mesh1d.elem[0].w_1v) = \n%r" % my_mesh1d.elem[0].dx.dot(my_mesh1d.elem[0].w_1v))
        print("my_mesh1d.gm = \n%r" % my_mesh1d.gm[0])
        my_mesh1d.plot()
    minimum_working_example()
