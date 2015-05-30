from nose.tools import *
import numpy

from solverls import spectral


__author__ = 'Alfredo Carella'


class TestSpectralLibrary(unittest.TestCase):

    def setUp(self):
        self.list_of_orders = list(range(2, 7))
        self.list_of_element_boundaries = [(-1, 1), (2, 5), (-6, -4), (-3, 1)]

    def tearDown(self):
        del self.list_of_orders
        del self.list_of_element_boundaries

    def number_of_points_and_boundary_coordinates(self):
        for order in self.list_of_orders:
            for (x_min, x_max) in self.list_of_element_boundaries:
                yield order + 1, (x_min, x_max)

    def test_gll(self):
        for number_of_points, (x_min, x_max) in self.number_of_points_and_boundary_coordinates():
            points, weights = spectral.gll(number_of_points, x_min, x_max)
            self.assertEqual(points[0], x_min)
            self.assertEqual(points[-1], x_max)
            assert_almost_equal(numpy.sum(weights), x_max - x_min)  # int(1)^{x_max}_{x_min}
            assert_almost_equal(weights.dot(points), (x_max**2 - x_min**2)/2.0)  # int(x)^{x_max}_{x_min}

    def test_lagrange_derivative_matrix_gll(self):
        for number_of_points, (x_min, x_max) in self.number_of_points_and_boundary_coordinates():
            points, weights = spectral.gll(number_of_points, x_min, x_max)
            dx = spectral.lagrange_derivative_matrix_gll(number_of_points) * 2.0 / (x_max - x_min)
            numpy.testing.assert_allclose(dx.dot(points), numpy.ones(number_of_points))  # d/dx(x)

    def test_lagrange_interpolating_matrix(self):
        for number_of_points, (x_min, x_max) in self.number_of_points_and_boundary_coordinates():
            points, weights = spectral.gll(number_of_points, x_min, x_max)
            for plot_resolution in [7, 12, 23]:
                plot_points = numpy.linspace(x_min, x_max, plot_resolution)
                l = spectral.lagrange_interpolating_matrix(points, plot_points)
                dx_matrix = spectral.lagrange_derivative_matrix_gll(len(points)) * 2.0 / (points[-1] - points[0])
                dx_in = dx_matrix.dot(points)
                numpy.testing.assert_allclose(l.dot(dx_in), numpy.ones(len(plot_points)))
                numpy.testing.assert_allclose(l.dot(points), plot_points)
