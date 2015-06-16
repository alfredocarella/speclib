import numpy

from solverls.lsproblem import LSProblem
from solverls.spectral import conj_grad_elem


__author__ = 'Alfredo Carella'


class LSProblemNonLinear(LSProblem):

    def solve(self, min_residual=1e-20, max_iter=50, min_delta=1e-16, seed=None):
        min_residual = min_residual
        max_iter = max_iter
        min_delta = min_delta

        delta = 1.0
        solution_iterations = 0
        has_converged = False
        is_converging = True
        reached_max_iter = False

        self.f = seed if seed else numpy.ones(self.mesh.dof)   # seed for guessing the solution
        while (not has_converged) and (not reached_max_iter) and is_converging:
            for el in self.mesh.elem:
                self.set_operators(el)
            self.set_boundary_conditions()
            self.f_old = self.f.copy()
            self.f, num_cg_it = conj_grad_elem(self.k_el, self.g_el, self.mesh.gm)
            self.residual = sum(self.compute_residual(el) for el in self.mesh.elem)
            solution_iterations += 1
            delta = numpy.linalg.norm(self.f - self.f_old) / numpy.linalg.norm(self.f)

            has_converged = self.residual < min_residual
            is_converging = delta > min_delta
            reached_max_iter = solution_iterations >= max_iter
            if not is_converging:
                raise RuntimeWarning("Equal consecutive nonlinear iterations. Delta = %r" % delta)
            elif reached_max_iter:
                raise RuntimeWarning("Reached max nonlinear iterations (%r)." % solution_iterations)

        solver_report = "Iterations: {0!r:s}  -  Residual: {1:04.2e}  -  delta = {2:04.2e}"
        print(solver_report.format(solution_iterations, self.residual, delta))


if __name__ == '__main__':
    pass
