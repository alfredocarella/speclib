import numpy

from solverls.lsproblem import LSProblem
from solverls.spectral import conj_grad

__author__ = 'alfredoc'


class LSProblemTimeMarching(LSProblem):
    def solve_linear_slab(self):
        self.f = numpy.zeros(self.mesh.dof)
        self.residual = 0
        for el in self.mesh.elem:
            self.set_operators(el)

            if el.number > 0:
                self.set_slab_boundary_conditions(el)
            else:
                self.set_boundary_conditions()

            f_elem, cg_iterations = conj_grad(self.k_el[0], self.g_el[0])
            self.f[el.nodes] = f_elem
            self.residual += self.compute_residual(el)

    def set_slab_boundary_conditions(self, el):
        for var in el.variables:
            f_index = el.nodes[el.pos[var]][0]
            gk_index = el.pos[var][0]
            self.k_el[0][gk_index, gk_index] += 1.0
            self.g_el[0][gk_index] += self.f[f_index]