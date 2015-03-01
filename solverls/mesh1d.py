import matplotlib.pyplot as plt
import numpy
from solverls.element1d import Element1D

__author__ = 'Alfredo Carella'


class Mesh1D(object):
    """
    This class represents a 1-dimensional high-order Finite Element mesh.
    macroGrid = vector with the position of element boundary nodes
    P = vector with the order for each element (int)
    NV = number of variables (int)
    """

    def __init__(self, macro_grid, element_orders, variable_names=('f',)):
        # Mesh attributes
        self.macro_grid = numpy.array(macro_grid, dtype=float)
        self.element_orders = numpy.atleast_1d(element_orders)
        self.variables = variable_names
        self.dof_1v = (numpy.sum(self.element_orders) + 1)
        self.dof = self.dof_1v * len(self.variables)
        # Gathering matrix
        self.gm = self.create_gm()
        # List of elements
        self.elem = []
        for el in range(len(self.macro_grid) - 1):
            self.elem.append(Element1D(self.macro_grid[el:el+2], self.element_orders[el], self.variables))
            self.elem[el].number = el
            self.elem[el].nodes = self.gm[el]

    def create_gm(self):
        gm = []
        for el in range(len(self.element_orders)):
            num_points = (self.element_orders[el] + 1)
            gm.append(numpy.zeros(len(self.variables) * num_points, dtype=numpy.int))
            for var in self.variables:
                var_index = self.variables.index(var)
                start_number = var_index * self.dof_1v + sum(self.element_orders[range(el)])
                span = numpy.arange(num_points)
                gm[-1][numpy.add(span, var_index * num_points)] = numpy.add(span, start_number)
        return gm

    def plot(self):
        # Plot nodes and element boundaries
        for el in self.elem:
            plt.plot((el.boundaries['x'][0], el.boundaries['x'][-1]), (0, 0), 'r--', linewidth=1.0)
            plt.plot(el.x_1v, el.x_1v * 0, 'ro')
            plt.plot(el.boundaries['x'], el.boundaries['x'] * 0, 'bs', markersize=10)
            plt.text(sum(el.boundaries['x']) / 2.0, +0.1, str(el.number), fontsize=15, color='blue')
            for node in range(el.order+1):
                plt.text(el.x_1v[node], -0.1, str(self.gm[el.number][node]), fontsize=10, color='red')

        # Write annotations
        macro_grid = self.macro_grid
        mesh_text = """
        Degrees of freedom: {0}
        Number of elements: {1}
        Variables: {2} {3}
        """.format(self.dof, len(self.elem), len(self.variables), self.variables)
        plt.text((macro_grid[0] + macro_grid[-1]) / 4.0, -0.9, mesh_text)
        plt.title('1-D mesh information')
        plt.xlabel('Independent variable coordinate')
        plt.axis([macro_grid[0], macro_grid[-1], -1, 1])
        plt.show()


if __name__ == '__main__':
    def minimum_mesh_example():
        macro_grid, orders, variables = [0.0, 1.0, 2.0, 3.0], [3, 4, 2], ['Temperature', 'Pressure', 'Quality']
        my_mesh1d = Mesh1D(macro_grid, orders, variables)

        print('Test inputs:')
        print('macro_grid = %s' % my_mesh1d.macro_grid)
        print('orders = %s' % my_mesh1d.element_orders)
        print('variables = %s' % my_mesh1d.variables)
        print('')
        print('Test outputs:')
        first_element = my_mesh1d.elem[0]
        print("my_mesh1d.elem[0].dx = \n%r" % first_element.dx)
        print("my_mesh1d.elem[0].x_1v = %r" % first_element.x_1v)
        print("my_mesh1d.elem[0].dx.dot(my_mesh1d.elem[0].w_1v) = \n%r" % first_element.dx.dot(first_element.w_1v))
        print("my_mesh1d.gm = \n%r" % my_mesh1d.gm[0])
        my_mesh1d.plot()
    minimum_mesh_example()
