__author__ = 'alfredoc'
from solverls.speclib import lagrange_derivative_matrix_gll, gll
import numpy

class Element1D():
    """Spectral (high order) one-dimensional element. One of the elementary blocks that compose a mesh (and a mesh object)"""
    #TODO: A suitable docstring should be written here

    def __init__(self, macro_nodes, gm_cell, list_of_variables):
        # self.quadrature_weights = []  # element attribute
        # self.quadrature_points = []  # element attribute
        # self.jac = []  # element attribute
        # self.dx = []  # element attribute
        # self.x = numpy.zeros(self.dof)  # element attribute
        # self.long_quadrature_weights = []  # element attribute
        # self.pos = []  # element attribute
        # for el_ in range(self.number_of_elements):
        #     lower_element_boundary = self.macro_nodes[el_]
        #     upper_element_boundary = self.macro_nodes[el_ + 1]
        #     self.jac.append((upper_element_boundary - lower_element_boundary) / 2.0)
        #     self.dx.append(lagrange_derivative_matrix_gll(self.element_orders[el_] + 1) / self.jac[el_])
        #     qx, qw = gll(self.element_orders[el_] + 1, lower_element_boundary, upper_element_boundary)
        #     self.quadrature_points.append(qx)  # x coordinate (quad points)
        #     self.quadrature_weights.append(qw)  # quadrature weights
        #     long_qw = numpy.tile(qw, self.number_of_variables)
        #     self.long_quadrature_weights.append(long_qw)
        #     self.pos.append({})
        #     for var_ in range(self.number_of_variables):
        #         first_node_in_element = var_ * len(self.gm[el_]) // self.number_of_variables
        #         last_node_in_element = first_node_in_element + len(self.gm[el_]) // self.number_of_variables
        #         self.pos[el_][self.list_of_variables[var_]] = numpy.arange(first_node_in_element, last_node_in_element)
        #     self.x[self.gm[el_]] = numpy.tile(qx, self.number_of_variables)

        self.number_of_variables = len(list_of_variables)
        self.element_order = (len(gm_cell) // self.number_of_variables) - 1
    # for el_ in range(self.number_of_elements):
        lower_element_boundary = macro_nodes[0]
        upper_element_boundary = macro_nodes[1]
        self.jac = (upper_element_boundary - lower_element_boundary) / 2.0
        self.dx = lagrange_derivative_matrix_gll(self.element_order + 1) / self.jac
        qx, qw = gll(self.element_order + 1, lower_element_boundary, upper_element_boundary)
        long_qw = numpy.tile(qw, self.number_of_variables)
        self.quadrature_points = qx  # x coordinate (quad points)
        self.quadrature_weights = qw  # quadrature weights
        self.long_quadrature_weights = long_qw
        self.pos = {}
        self.x[gm_cell] = numpy.tile(qx, self.number_of_variables)
        for var_ in range(self.number_of_variables):
            first_node_in_variable = var_ * (self.element_order + 1)
            last_node_in_variable = first_node_in_variable + (self.element_order + 1)
            self.pos[list_of_variables[var_]] = numpy.arange(first_node_in_variable, last_node_in_variable)


