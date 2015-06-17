import numpy

from solverls.lsproblem import LSProblem
from solverls.spectral import conj_grad_elem


__author__ = 'Alfredo Carella'


class LSProblemNonLinear(LSProblem):

    def solve(self, max_nonlinear_iterations=50, nonlinear_tolerance=1e-8, seed=None):
        nonlinear_iterations = 0
        has_converged = False
        self.f_old = seed if seed else numpy.ones(self.mesh.dof)   # seed for guessing the solution
        while (not has_converged) and (nonlinear_iterations < max_nonlinear_iterations):
            nonlinear_iterations += 1
            self.set_problem()
            self.f, num_cg_it = conj_grad_elem(self.k_el, self.g_el, self.mesh.gm)
            solution_variation = numpy.linalg.norm(self.f - self.f_old) / numpy.linalg.norm(self.f)
            self.f_old = self.f.copy()
            has_converged = solution_variation < nonlinear_tolerance

            if nonlinear_iterations == max_nonlinear_iterations:
                raise RuntimeWarning("Reached max number of nonlinear iterations (%s)." % nonlinear_iterations)

        self.residual = sum(self.compute_residual(el) for el in self.mesh.elem)


if __name__ == '__main__':
    pass
