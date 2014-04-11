import numpy
from solverls.speclib import conj_grad, conj_grad_elem

__author__ = 'raul'


class Iterator(object):

    def __init__(self, min_residual=1e-20, max_nonlinear_it=50, min_delta=1e-16):
        self.min_residual = min_residual
        self.max_nonlinear_it = max_nonlinear_it
        self.min_delta = min_delta

        self.delta = 1.0
        self.number_of_iterations = 0
        self.converged = False
        self.converging = True
        self.reached_max_it = False

    def iterate(self, problem, set_operators, set_boundary_conditions, list_of_elements=None):
        while (not self.converged) and (self.converging) and (not self.reached_max_it):

            set_operators(list_of_elements)
            set_boundary_conditions()
            problem.f_old = problem.f.copy()

            if len(problem.mesh.gm[0]) == problem.mesh.dof_nv:
                problem.f, num_cg_it = conj_grad(problem.k_el[0], problem.g_el[0])
            else:
                problem.f, num_cg_it = conj_grad_elem(problem.k_el, problem.g_el, problem.mesh.gm, problem.mesh.dof_nv)

            problem.residual = problem.compute_residual(list_of_elements)
            self.delta = numpy.linalg.norm(problem.f - problem.f_old) / numpy.linalg.norm(problem.f)

            self.number_of_iterations += 1

            if problem.residual < self.min_residual:
                # print("Converged: residual below tolerance. Residual < %r" % it.MIN_DELTA)
                self.converged = True
            elif self.delta < self.min_delta:
                print("Equal consecutive nonlinear iterations. Delta = %r" % self.delta)
                self.converging = False
            elif self.number_of_iterations >= self.max_nonlinear_it:
                print("Stopping after having reached %r nonlinear iterations." % self.number_of_iterations)
                self.reached_max_it = True
            else:
                pass