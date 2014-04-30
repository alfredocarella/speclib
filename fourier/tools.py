__author__ = 'raul'

import math
import numpy

def harmonics_to_time(frequencies, amplitudes, phases, time_coord):
    time_signal = numpy.zeros(time_coord.shape)
    for index in range(len(frequencies)):
        freq = frequencies[index]
        amplitude = amplitudes[index]
        phase = math.radians(phases[index])
        time_signal += amplitude * numpy.cos(2*math.pi * freq * time_coord + phase)

    return time_signal