__author__ = 'raul'

# First attempt on "test driven development".
# - The concept works (1st failing test; 2nd working function)
# - Functionality is still useless

from fourier.tools import harmonics_to_time
import numpy
import numpy.testing
import math

def test_harmonics_to_time():
    frequencies = numpy.array([0, 1])
    amplitudes = numpy.array([1, 2])
    phases = numpy.array([0, 0])
    time_coordinates = numpy.linspace(0, 2, num=50, endpoint=True)

    time_domain_vector = harmonics_to_time(frequencies, amplitudes, phases, time_coordinates)
    expected_result = 1 + 2*numpy.cos(2*math.pi * time_coordinates)
    numpy.testing.assert_allclose(time_domain_vector, expected_result)
