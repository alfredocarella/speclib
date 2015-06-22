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
        self.pos = self.create_variable_indices()
        # Geometry
        self.boundaries = {'x': boundaries}
        self.x_1v, self.w_1v = gll(self.order, *boundaries)
        self.w_nv = numpy.tile(self.w_1v, len(variables))
        self.x_nv = numpy.tile(self.x_1v, len(variables))
        # Differentiation
        self.jac = (boundaries[1] - boundaries[0]) / 2.0
        self.dx = gll_derivative_matrix(self.order) / self.jac

    def create_variable_indices(self):
        pos = {}
        dof_local = (self.order + 1)
        for idx, var in enumerate(self.variables):
            pos[var] = numpy.arange(idx * dof_local, (idx + 1) * dof_local)
        return pos
