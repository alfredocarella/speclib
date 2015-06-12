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
        for idx, order in enumerate(self.element_orders):
            self.elem.append(Element1D(self.macro_grid[idx:idx + 2], order, self.variables))
            self.elem[idx].number = idx
            self.elem[idx].nodes = self.gm[idx]

    def create_gm(self):
        gm = []
        node_count_for_first_variable = 0
        for idx_el, order in enumerate(self.element_orders):
            num_points = (order + 1)
            gm.append(numpy.zeros(len(self.variables) * num_points, dtype=numpy.int))
            for idx_var, var in enumerate(self.variables):
                start_number = idx_var * self.dof_1v + node_count_for_first_variable
                span = numpy.arange(num_points)
                gm[-1][numpy.add(span, idx_var * num_points)] = numpy.add(span, start_number)
            node_count_for_first_variable += order
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
