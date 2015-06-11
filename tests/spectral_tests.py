import itertools

import nose
import numpy

from solverls import spectral

__author__ = 'Alfredo Carella'


def spectral_test_case_generator():
    tested_orders = range(2, 7)
    tested_boundaries = [(-1, 1), (2, 5), (-6, -4), (-3, 1)]
    for order, (x_min, x_max) in itertools.product(*[tested_orders, tested_boundaries]):
        points, weights = spectral.gll(order, x_min, x_max)
        yield check_all_quadrature_functions, points, weights


def check_all_quadrature_functions(points, weights):
    # Testing spectral.gll()
    nose.tools.assert_almost_equal(numpy.sum(weights), points[-1] - points[0],
                                   msg="int(1)^{x_max}_{x_min} != x_max - x_min")
    nose.tools.assert_almost_equal(weights.dot(points), (points[-1] ** 2 - points[0] ** 2) / 2.0,
                                   msg="int(x)^{x_max}_{x_min} != (x_max^2 - x_min^2)/2")

    # Testing spectral.gll_derivative_matrix()
    jacobian = 2.0 / (points[-1] - points[0])
    dx = spectral.gll_derivative_matrix(len(points) - 1) * jacobian
    numpy.testing.assert_allclose(dx.dot(points), numpy.ones(len(points)), err_msg="d/dx(x) != 1")
    numpy.testing.assert_almost_equal(dx.dot(points**2.0), 2.0 * points, decimal=10, err_msg="d/dx(x^2) != x")

    # Testing spectral.interpolant_evaluation_matrix()
    for plot_resolution in [7, 12, 23]:
        plot_points = numpy.linspace(points[-1], points[0], plot_resolution)
        l = spectral.interpolant_evaluation_matrix(points, plot_points)
        dx_in = dx.dot(points)
        numpy.testing.assert_allclose(l.dot(dx_in), numpy.ones(len(plot_points)))
        numpy.testing.assert_allclose(l.dot(points), plot_points)
        numpy.testing.assert_almost_equal(l.dot(points ** 2.0), plot_points ** 2.0)
