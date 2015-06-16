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
