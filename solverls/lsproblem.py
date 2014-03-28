import os
import matplotlib.pyplot
import numpy
import pylab
from solverls.speclib import info, lagrange_interpolating_matrix, conj_grad_elem, Iterator, conj_grad

__author__ = 'raul'


class LSProblem(object):

    def __init__(self, theMesh):
        self.mesh = theMesh

    def computeResidual(self, list_of_elements=None):
        """Compute the Least-Squares total residual."""

        if list_of_elements is None:
            list_of_elements = range(len(self.mesh.gm))

        residual = 0.0
        for el_ in list_of_elements:
            W = self.mesh.long_quadrature_weights[el_]
            gm = self.mesh.gm[el_]
            opG = self.opG[el_]
            opL = self.opL[el_]
            residual += W.dot((opL.dot(self.f[gm])-opG)**2)  # Int[(Lf-G)**2] + [f_a-f(a)]**2   <---The second term is missing. Therefore this formulation does not currently consider compliance with boundary conditions!
        return residual

    def plotSolution(self, varList=None, filename=None):  # TODO: This routine still needs to be polished and documented

        if varList == None: varList = self.mesh.list_of_variables

        fig = matplotlib.pyplot.figure()
        for varName in varList:
            varNumber = self.mesh.list_of_variables.index(varName)
            xIn = numpy.zeros([])
            yIn = numpy.zeros([])
            xOut = numpy.zeros([])
            yOut = numpy.zeros([])
            for elem in range(self.mesh.number_of_elements):
                varIndices = self.mesh.gm[elem][self.mesh.pos[elem][varName]]
                xInElem = self.mesh.x[varIndices]
                yInElem = self.f[varIndices]
                xOutElem = numpy.linspace(self.mesh.macro_nodes[elem], self.mesh.macro_nodes[elem+1], 10)
                yOutElem = lagrange_interpolating_matrix(xInElem, xOutElem).dot(yInElem)
                xIn = numpy.append(xIn, xInElem)
                yIn = numpy.append(yIn, yInElem)
                xOut = numpy.append(xOut, xOutElem)
                yOut = numpy.append(yOut, yOutElem)

            matplotlib.pyplot.subplot(100*len(varList)+10+varNumber)
            matplotlib.pyplot.plot(xOut, yOut, '--', linewidth=2.0)
            matplotlib.pyplot.plot(xIn, yIn, '.', markersize=8.0)
            matplotlib.pyplot.ylabel(varName)
            matplotlib.pyplot.xlabel('x (independent variable)')

        if filename != None:
            if not os.path.exists('output'):
                os.makedirs('output')
            pylab.savefig('output//'+filename, bbox_inches=0)
            print("Functions %r have been printed to file '%s'" %(varList, filename))
        else:
            matplotlib.pyplot.show()

        return fig

    def setBoundaryConditions(self):
        info("THIS FUNCTION MUST BE OVERRIDEN IN CHILD CLASS")  # FIXME: Use an error instead of the "info" function

    def setSlabBoundaryConditions(self, elem):
        weight = 1.0
        for varName in self.mesh.list_of_variables:

            finalValueIndices = self.mesh.gm[elem-1][self.mesh.pos[elem-1][varName]]
            initialValueIndices = self.mesh.pos[elem][varName]
            f_index = finalValueIndices[-1]
            gk_index = initialValueIndices[0]

            self.Ke[0][gk_index, gk_index] += weight
            self.Ge[0][gk_index] += weight * self.f[f_index]

    def setEquations(self, el):
        info("THIS FUNCTION MUST BE OVERRIDEN IN CHILD CLASS")  # FIXME: Use an error instead of the "info" function

    def setOperators(self, list_of_elements=None):

        if list_of_elements is None:
            list_of_elements = range(self.mesh.number_of_elements)
        else:
            list_of_elements = list(list_of_elements)

        self.opL = []
        self.opG = []
        self.Ke = []
        self.Ge = []

        # Create the operators
        for elem in list_of_elements:
            el_ = list_of_elements.index(elem)
            elemSize = len(self.mesh.gm[elem])
            self.opL.append( numpy.zeros((elemSize, elemSize)) )
            self.opG.append( numpy.zeros((elemSize)) )

            opL_dict, opG_dict = self.setEquations(elem)

            for varRow_ in self.mesh.list_of_variables:
                for varCol_ in self.mesh.list_of_variables:
                    if (varRow_+'.'+varCol_) in opL_dict:
                        self.opL[el_][numpy.ix_(self.mesh.pos[elem][varRow_], self.mesh.pos[elem][varCol_])] += opL_dict[varRow_+'.'+varCol_]
                if varRow_ in opG_dict:
                    self.opG[el_][self.mesh.pos[elem][varRow_]] += opG_dict[varRow_]

            # Generate problem sub-matrices
            elemNodes = self.mesh.gm[elem]
            LW = self.opL[el_].T.dot(numpy.diag(self.mesh.long_quadrature_weights[elem]))
            self.Ke.append(LW.dot(self.opL[el_]))
            self.Ge.append(LW.dot(self.opG[el_]))

    def setSolution(self, f):
        self.fOld = f

    def solveLinear(self):
        # it = Iterator(MIN_RESIDUAL=1e-20, MAX_NONLINEAR_ITERS=50, MIN_DELTA=1e-16)
        # self.f = self.mesh.x    # or even --> numpy.ones(len(self.mesh.x))
        # it.iteratePicard(self, self.setOperators, self.setBoundaryConditions)
        self.setOperators()
        self.setBoundaryConditions()
        self.f, numIters = conj_grad_elem(self.Ke, self.Ge, self.mesh.gm, self.mesh.dof_nv)

    def solveNonLinear(self):
        self.f = self.mesh.x    # or even --> numpy.ones(len(self.mesh.x))
        it = Iterator(min_residual=1e-20, max_nonlinear_it=50, min_delta=1e-16)
        it.iteratePicard(self, self.setOperators, self.setBoundaryConditions, [0,1])

        print("Iterations: %r  -  Residual: %04.2e  -  delta = %04.2e" % (it.number_of_iterations, self.residual, it.delta))

    def solveLinearSlab(self):
        self.f = numpy.zeros(self.mesh.dof_nv)

        self.setOperators([0])
        self.setBoundaryConditions()
        f_elem, numIters = conj_grad(self.Ke[0], self.Ge[0])
        self.f[self.mesh.gm[0]] = f_elem

        for el_ in range(1,self.mesh.number_of_elements):
            self.setOperators([el_])
            self.setSlabBoundaryConditions(el_)
            f_elem, numIters = conj_grad(self.Ke[0], self.Ge[0])
            self.f[self.mesh.gm[el_]] = f_elem

    def solveNonLinearSlab(self):  # TODO: Would be very useful, but it is complex to implement right now
        pass