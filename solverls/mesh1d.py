import matplotlib.pyplot
import numpy
from solverls.speclib import lagrange_derivative_matrix_gll, gll

__author__ = 'raul'


class Mesh1d(object):
    """
    This class represents a 1-dimensional high-order Finite Element mesh.
    macroGrid = vector with the position of element boundary nodes
    P = vector with the order for each element (int)
    NV = number of variables (int)

    The attributes of class Mesh1d()are:

    Mesh1d.macroNodes
    Mesh1d.elementOrders
    Mesh1d.numberOfElements
    Mesh1d.listOfVariables
    Mesh1d.numberOfVariables
    Mesh1d.dof1v
    Mesh1d.dof_nv
    Mesh1d.gm
    Mesh1d.gm_1v
    Mesh1d.quadWeights
    Mesh1d.quadPoints
    Mesh1d.Jx
    Mesh1d.Dx
    Mesh1d.x
    Mesh1d.longQuadWeights

    Example:
    macroGrid, P, NV = numpy.array((0.0,1.0,2.0,3.0)), numpy.array((3,4,2)), 2
    myMesh1d = Mesh1d(macroGrid, P, NV)
    print("myMesh1d.Dx[0] = \n%r" % myMesh1d.Dx[0])
    print("myMesh1d.x[myMesh1d.gm1v[0]] = %r" % myMesh1d.x[myMesh1d.gm_1v[0]])
    print("myMesh1d.Dx[0].dot(myMesh1d.quadPoints[0]) = \n%r\" % myMesh1d.Dx[0].dot(myMesh1d.quadPoints[0]))
    """

    def __init__(self, macro_grid, element_orders, variable_names=['f']):
        self.macro_nodes = macro_grid
        self.element_orders = numpy.atleast_1d(element_orders)
        self.number_of_elements = len(self.element_orders)
        self.list_of_variables = variable_names
        self.number_of_variables = len(self.list_of_variables)
        self.dof_1v = numpy.sum(self.element_orders)+1
        self.dof_nv = self.dof_1v * self.number_of_variables

        self.gm = []
        self.gm_1v = []
        node_counter = 0
        for el_ in range(self.number_of_elements):
            element_size = self.number_of_variables * (self.element_orders[el_]+1)
            self.gm.append(numpy.zeros(element_size, dtype=numpy.int))
            for var_ in range(self.number_of_variables):
                first_node_position = var_ * (self.element_orders[el_]+1)
                last_node_position = first_node_position + self.element_orders[el_]
                first_node_number = var_ * self.dof_1v + node_counter
                last_node_number = first_node_number + self.element_orders[el_]
                self.gm[el_][first_node_position: last_node_position + 1] = numpy.arange(first_node_number, last_node_number + 1)
            node_counter += self.element_orders[el_]
            self.gm_1v.append(self.gm[el_][0: self.element_orders[el_]+1])

        self.quadrature_weights = []
        self.quadrature_points = []
        self.jac = []
        self.dx = []
        self.x = numpy.zeros(self.dof_1v)
        self.long_quadrature_weights = []
        self.pos = []
        for el_ in range(self.number_of_elements):
            lower_element_boundary = self.macro_nodes[el_]
            upper_element_boundary = self.macro_nodes[el_+1]
            self.jac.append((upper_element_boundary - lower_element_boundary) / 2.0)
            self.dx.append(lagrange_derivative_matrix_gll(self.element_orders[el_]+1) / self.jac[el_])
            qx, qw = gll(self.element_orders[el_]+1, lower_element_boundary, upper_element_boundary)
            self.quadrature_points.append(qx)    # x coordinate (quad points)
            self.quadrature_weights.append(qw)    # quadrature weights
            long_qw = numpy.tile(qw, self.number_of_variables)
            self.long_quadrature_weights.append(long_qw)
            self.pos.append({})
            for var_ in range(self.number_of_variables):
                first_node_in_element = var_ * len(self.gm_1v[el_])
                last_node_in_element = first_node_in_element + len(self.gm_1v[el_])
                self.pos[el_][self.list_of_variables[var_]] = numpy.arange(first_node_in_element, last_node_in_element)
            self.x[self.gm_1v[el_]] = qx

        self.x = numpy.tile(self.x, self.number_of_variables)

    def plot_mesh(self):
        # Plot nodes and line
        micro_grid = self.x
        macro_grid = self.macro_nodes
        matplotlib.pyplot.plot((macro_grid[0], macro_grid[-1]), (0, 0), 'r--', linewidth=2.0)    # Lines
        matplotlib.pyplot.plot(micro_grid, micro_grid*0, 'ro')    # Nodes (micro)
        matplotlib.pyplot.plot(macro_grid, macro_grid*0, 'bs', markersize=10)    # Nodes (macro)

        # Plot node and element numbers
        for node_ in range(self.dof_1v):
            matplotlib.pyplot.text(self.x[node_], -0.1, str(node_), fontsize=10, color='red')
        for border_ in range(len(macro_grid)-1):
            element_center = (macro_grid[border_] + macro_grid[border_+1]) / 2.0
            matplotlib.pyplot.text(element_center, +0.1, str(border_), fontsize=15, color='blue')

        # Write annotations
        first_element_center = ( macro_grid[0] + macro_grid[1] ) / 2.0
        matplotlib.pyplot.annotate('element numbers', xy=(first_element_center, 0.17), xytext=(first_element_center, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
        matplotlib.pyplot.annotate('node number', xy=(micro_grid[1], -0.12), xytext=(micro_grid[1], -0.3), arrowprops=dict(facecolor='black', shrink=0.05))
        matplotlib.pyplot.text((macro_grid[0]+macro_grid[-1])/4.0, -0.9, 'Degrees of freedom per variable: %d\nTotal degrees of freedom: %d\nNumber of elements: %d\nVariables: %d %s' %(self.dof_1v, self.dof_nv, self.number_of_elements, self.number_of_variables, self.list_of_variables))
        matplotlib.pyplot.title('1-D mesh information')
        matplotlib.pyplot.xlabel("Independent variable coordinate")
        matplotlib.pyplot.axis([macro_grid[0], macro_grid[-1], -1, 1])
        matplotlib.pyplot.show()