from solverls.lsproblem import LSProblem
from solverls.spectral import conj_grad_elem

__author__ = 'Alfredo Carella'


class LSProblemLinear(LSProblem):

    def solve(self):
        for el in self.mesh.elem:
            self.set_operators(el)
        self.set_boundary_conditions()
        self.f, cg_iterations = conj_grad_elem(self.k_el, self.g_el, self.mesh.gm)
        self.residual = sum(self.compute_residual(el) for el in self.mesh.elem)


if __name__ == '__main__':
    pass
