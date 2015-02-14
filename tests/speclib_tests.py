from nose.tools import *
import numpy

from solverls import speclib


def test_gll(number_of_points=5, x_min=-1, x_max=1):
        points, weights = speclib.gll(number_of_points, x_min, x_max)
        assert_equal(points[0], x_min)
        assert_equal(points[-1], x_max)
        assert_almost_equal(numpy.sum(weights), x_max - x_min)


def test_lagrange_derivative_matrix_gll(number_of_points=2, x_min=-1, x_max=1, points=[-1, 1]):
        dx = speclib.lagrange_derivative_matrix_gll(number_of_points) * 2.0/(x_max-x_min)
        dp = dx.dot(points)
        numpy.testing.assert_allclose(dp, numpy.ones(number_of_points))


def test_lagrange_interpolating_matrix(x_in=[-1, 1], x_out=[-1, 0, 1]):
        l = speclib.lagrange_interpolating_matrix(x_in, x_out)
        dx_matrix = speclib.lagrange_derivative_matrix_gll(len(x_in)) * 2.0/(x_in[-1]-x_in[0])
        dx_in = dx_matrix.dot(x_in)
        numpy.testing.assert_allclose(l.dot(dx_in), numpy.ones(len(x_out)))


for n in range(2, 5):
    x_left, x_right = 0, 4
    plot_res = 30
    test_gll(n, x_left, x_right)
    p, w = speclib.gll(n, x_left, x_right)
    test_lagrange_derivative_matrix_gll(n, x_left, x_right, p)
    x_plot = numpy.linspace(x_left, x_right, plot_res)
    test_lagrange_interpolating_matrix(p, x_plot)