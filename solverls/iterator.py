import numpy
from solverls.speclib import conj_grad, conj_grad_elem

__author__ = 'Alfredo Carella'


class Iterator(object):

    def __init__(self, min_residual=1e-20, max_nonlinear_it=50, min_delta=1e-16):
        self.min_residual = min_residual
        self.max_nonlinear_it = max_nonlinear_it
        self.min_delta = min_delta

        self.delta = 1.0
        self.solution_iterations = 0
        self.converged = False
        self.converging = True
        self.reached_max_it = False

    def iterate(self, problem):
        while (not self.converged) and (not self.reached_max_it) and self.converging:

            problem.set_operators()
            problem.set_boundary_conditions()
            problem.f_old = problem.f.copy()

            problem.f, num_cg_it = conj_grad_elem(problem.k_el, problem.g_el, problem.mesh.gm, problem.mesh.dof)
            self.solution_iterations += 1

            problem.residual = problem.compute_residual()
            self.delta = numpy.linalg.norm(problem.f - problem.f_old) / numpy.linalg.norm(problem.f)

            self.converged = problem.residual < self.min_residual
            self.converging = self.delta > self.min_delta
            self.reached_max_it = self.solution_iterations >= self.max_nonlinear_it

            if not self.converging:
                raise RuntimeWarning("Equal consecutive nonlinear iterations. Delta = %r" % self.delta)
            elif self.reached_max_it:
                raise RuntimeWarning("Stopping after having reached %r nonlinear iterations." % self.solution_iterations)