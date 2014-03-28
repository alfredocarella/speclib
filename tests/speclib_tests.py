from nose.tools import *
from solverls import speclib

import numpy
from solverls.lsproblem import LSProblem


def test_math():    # Testing some of the mathematical routines
    """
    # The dependency-tree 'speclib' is:
    #
    # *GLL(np, x_min, x_max)
    #     *gaussLobattoLegendre(np)
    #         *gaussLegendre(np)
    #             *legendreDerivative(n,x)
    #         *legendreDerivative(n, x)
    #         *legendrePolynomial(n, x)
    # *lagrangeDerivativeMatrixGLL(np)
    #     *gaussLobattoLegendre(np)
    #     *legendrePolynomial(n, x)
    # *lagrangeInterpolantMatrix(xIn, xOut)
    #
    # I automatically test only the higher level functions, assuming that if they are OK, then the functions used by them are also OK
    """
    
    np=5
    x_min, x_max = 0, 4
    nInterp = 30
    delta = x_max - x_min
    
    # speclib.GLL(np, x_min, x_max) tested here
    p, w = speclib.gll(np, x_min, x_max)
    assert_equal(p[0], x_min)
    assert_equal(p[-1], x_max)
    assert_almost_equal(numpy.sum(w), (delta))
    
    # speclib.lagrangeDerivativeMatrixGLL(np) tested here    
    Dx = speclib.lagrange_derivative_matrix_gll(np) * 2.0/delta
    Dp = Dx.dot(p)
    numpy.testing.assert_allclose(Dp, numpy.ones(np))

    # speclib.lagrangeInterpolantMatrix(x_in, x_out) tested here    
    x_in = p
    x_out = numpy.linspace(x_min, x_max, nInterp)
    L = speclib.lagrange_interpolating_matrix(x_in, x_out)
    numpy.testing.assert_allclose(L.dot(Dp), numpy.ones(nInterp))
    
    speclib.info("Execution complete!")
    

def test_mesh1d():    # Testing the mesh generation (plotting is not tested here)
    macroGrid, P, varList = numpy.array((0.0,1.0,2.0,3.0)), numpy.array((3,4,2)), ['T', 'pres', 'quality']    
    myMesh1d = speclib.Mesh1d(macroGrid, P, varList)
    
    numpy.testing.assert_array_equal(myMesh1d.element_orders, P)
    numpy.testing.assert_allclose(myMesh1d.macro_nodes, macroGrid)
    assert_equal(myMesh1d.list_of_variables, varList)
    assert_equal(myMesh1d.number_of_elements, len(myMesh1d.element_orders))
    assert_equal(myMesh1d.number_of_variables, len(myMesh1d.list_of_variables))
    assert_equal(myMesh1d.dof1v, sum(myMesh1d.element_orders)+1)
    assert_equal(myMesh1d.dofNv, myMesh1d.number_of_variables*myMesh1d.dof1v)
    assert_equal(len(myMesh1d.GM), myMesh1d.number_of_elements)
    assert_equal(len(myMesh1d.GM1v), myMesh1d.number_of_elements)

    integralTestValue = 0
    for el in range(myMesh1d.number_of_elements):
        integralTestValue += myMesh1d.quadrature_points[el].dot(myMesh1d.quadrature_weights[el])
        assert_almost_equal(myMesh1d.Jx[el], (myMesh1d.x[myMesh1d.GM1v[el][-1]]-myMesh1d.x[myMesh1d.GM1v[el][0]])/2.0 )
        numpy.testing.assert_array_equal(myMesh1d.GM[el][:myMesh1d.element_orders[el]+1], myMesh1d.GM1v[el])
        numpy.testing.assert_array_equal(myMesh1d.quadrature_points[el], myMesh1d.x[myMesh1d.GM1v[el]])
        numpy.testing.assert_array_equal(myMesh1d.Dx[el], speclib.lagrange_derivative_matrix_gll(myMesh1d.element_orders[el]+1)/myMesh1d.Jx[el])
        numpy.testing.assert_array_equal(myMesh1d.long_quadrature_weights[el], numpy.tile(myMesh1d.quadrature_weights[el], myMesh1d.number_of_variables))
        posVarTestValue = []
        for var in myMesh1d.list_of_variables:
            posVarTestValue = numpy.append(posVarTestValue, myMesh1d.pos[el][var])
        numpy.testing.assert_array_equal(posVarTestValue, range(len(myMesh1d.GM[el])))
    assert_almost_equal(integralTestValue, (myMesh1d.macro_nodes[-1]-myMesh1d.macro_nodes[0])**2 /2)
        
    
def testingProblem1el1v():    # Testing results for a simple problem (1 var, 1 elem)
    macroGrid, P, varList = numpy.array((0.0,2.0)), numpy.array((4)), ['f']
    myMesh1d = speclib.Mesh1d(macroGrid, P, varList)

    myProblem = LSProblemChildTest1el1v(myMesh1d)

    myProblem.residual = myProblem.computeResidual()
    assert_almost_equal(myProblem.residual, 0)
    # numpy.testing.assert_allclose(myProblem.Ke, myProblem.opL[0].T.dot(numpy.diag(myProblem.mesh.longQuadWeights[0])).dot(myProblem.opL[0])) #<--Missing BCs

    print("myProblem.opL = %r" % myProblem.opL)
    print("myProblem.opG = %r" % myProblem.opG)
    print("myProblem.Ke = %r" % myProblem.Ke)
    print("myProblem.Ge = %r" % myProblem.Ge)

    print("The residual for this problem is %04.2e" % myProblem.residual)
    print('\nThe solution vector is %r\n' % (myProblem.f))

    myMemo = """
    2013-11-18: A MINIMUM EXAMPLE IS WORKING!!! :-)

    Check-list for project (pending tasks):
    -  Add support for multi-equation
    -  Add support for multi-element
    -  Try first mechanical problem
    -  Solve element by element
    - self.computeResidual(self) <--Compliance with BC not considered yet!
    """

    print("Execution complete!\n\n"+myMemo)
#<only residual was tested>

# THE FOLLOWING AUTOMATIC ('assert'-like) TESTS ARE YET TO BE WRITTEN

def testingProblemNelNv():    # Testing a problem w/ multiple variables and elements
    macroGrid, P, varList = numpy.array((0.0, 1.0, 2.0)), numpy.array((3, 3)), ['f', 'g']
    print("macroGrid = %r - P = %r - varList = %r" % (macroGrid, P, varList))
    myMesh1d = speclib.Mesh1d(macroGrid, P, varList)
    print("myMesh1d = Mesh1d(macroGrid, P, varList)")

    myProblem = LSProblemChildTestNelNv(myMesh1d)
    myProblem.residual = myProblem.computeResidual()
    myProblem.plotSolution(['f','g'], 'testingProblemNelNv.pdf')

    # print("myProblem.opL = %r" % myProblem.opL)
    # print("myProblem.opG = %r" % myProblem.opG)
    # print("myProblem.mesh.Dx = %r" % myProblem.mesh.Dx)
    # print("myProblem.mesh.GM = %r" % myProblem.mesh.GM)

    # print('\nThe "elemGM" solution vector is %r\n' % (myProblem.f))
    print("The residual for this problem is %04.2e" % myProblem.residual)

    myMemo = """
    2013-11-27: A MINIMUM EXAMPLE IS WORKING!!! :-)

    Check-list for project (pending tasks):
    -  Add support for multi-equation *Done!
    -  Add support for multi-element *Done!
    -  Solve element by element *Done!
    -  Plot problem solutions *Done! (good enough now, can be improved later)
    -  Implement a solver routine for non-linear systems (Picard iter, consider Newton also)
    -  Implement a time-slab approach for only one time element at the time
    -  Try first mechanical problem (N variables with a self-generated system of equations)
    -  Consider showing animations (rotating masses / springs) before showing
    -  Consider including a steady-state Fourier analysis module
    - self.computeResidual(self) <--Compliance with BC not considered yet!
    - Find out how to do a code profiling
    """

    print(myMemo + '\n' + "testingProblemNelNv(): Execution complete!")

def testingProblemNonLinear():    # Testing iterative routine for solving a non-linear problem
    macroGrid, P, varList = numpy.array((0.0, 1.0, 2.0)), numpy.array((3, 3)), ['f']
    print("macroGrid = %r - P = %r - varList = %r" % (macroGrid, P, varList))

    myMesh1d = speclib.Mesh1d(macroGrid, P, varList)
    myProblem = NonLinearProblemTest(myMesh1d)
    myProblem.plotSolution(['f'], 'testingProblemNonLinear.pdf')

    print("The residual for this problem is %04.2e" % myProblem.residual)

    myMemo = """
    2013-12-02: A MINIMUM EXAMPLE IS WORKING!!! :-)

    Check-list for project (pending tasks):
    -  Add support for multi-equation *Done!
    -  Add support for multi-element *Done!
    -  Solve element by element *Done!
    -  Plot problem solutions *Done! (good enough now, can be improved later)
    -  Implement a solver routine for non-linear systems *Done! (Picard iteration is working)
    -  Implement a time-slab approach for only one time element at the time
    -  Try first mechanical problem (N variables with a self-generated system of equations)
    -  Consider showing animations (rotating masses / springs) before showing
    -  Consider including a steady-state Fourier analysis module
    - self.computeResidual(self) <--Compliance with BC not considered yet!
    - Find out how to do a code profiling
    """

    speclib.info("Execution complete!" + '\n' + myMemo)

def testingProblemTorsional1v():    # Testing a torsional vibration problem (1 mass)
    macroGrid = numpy.linspace(0.0, 30.0, 50)
    P = [4] * (len(macroGrid)-1)
    varList = ['v0', 'x0']
    # print("macroGrid = %r - P = %r - varList = %r" % (macroGrid, P, varList))

    myMesh1d = speclib.Mesh1d(macroGrid, P, varList)
    myProblem = TorsionalProblemTest(myMesh1d)
    myProblem.plotSolution()#filename='testingProblemTorsional1v.pdf')

    speclib.info("'TorsionalProblemTest.computeResidual()' does not work.")
    # # The following line will not work because myProblem.opL and and myProblem.opG have been reduced to 1 element. The full info is not saved
    # print("The residual for this problem is %04.2e" % myProblem.computeResidual())

    myMemo = """
    2013-12-04: A MINIMUM EXAMPLE IS WORKING!!! :-)

    Check-list for project (pending tasks):
    -  DONE! Add support for multi-equation
    -  DONE! Add support for multi-element
    -  DONE! Solve element by element
    -  DONE! Plot problem solutions (good enough now, can be improved later)
    -  DONE! Implement a solver routine for non-linear systems (Picard iteration is working)
    -  DONE! Implement a time-slab approach for only one time element at the time
    -  Try first mechanical problem (N variables with a self-generated system of equations)
    -  Consider creating animations (rotating masses / springs) before presenting script
    -  Consider including a steady-state Fourier analysis module
    - self.computeResidual(self) <--Compliance with BC not considered yet!
    - Find out how to do a code profiling
    """

    print(myMemo + '\n' + "testingProblemTorsional1v(): Execution complete!")

def testingProblemTorsionalNv():    # Testing a torsional vibration problem (N masses)
    macroGrid = numpy.linspace(0.0, 30.0, 40)
    P = [5] * (len(macroGrid)-1)
    numberOfMasses = 2

    varList = []
    for varNum in range(numberOfMasses):
        varList.append('v%d' % varNum)
        varList.append('x%d' % varNum)
    print(varList)

    myMesh1d = speclib.Mesh1d(macroGrid, P, varList)
    myProblem = TorsionalProblemTestNv(myMesh1d)
    myProblem.solveLinearSlab()
    myProblem.plotSolution()#filename='testingProblemTorsionalNv.pdf')

    speclib.info("'TorsionalProblemTestNv.computeResidual()' does not work.")
    # # The following line will not work because myProblem.opL and and myProblem.opG have been reduced to 1 element. The full info is not saved
    # print("The residual for this problem is %04.2e" % myProblem.computeResidual())

    myMemo = """
    2013-12-??: A MINIMUM EXAMPLE IS WORKING!!! :-)

    Check-list for project (pending tasks):
    -  Implement a non-linear time-slab approach
    -  Try first mechanical problem (N variables with a self-generated system of equations)
    -  Consider creating animations (rotating masses / springs) before presenting script
    -  Consider including a steady-state Fourier analysis module
    - self.computeResidual(self) <--Compliance with BC not considered yet!
    - Find out how to do a code profiling
    """

    print(myMemo + '\n' + "testingProblemTorsionalNv(): Execution complete!")

    print("range(1,1) = %r" % range(1,1))



# ********************************************************** #
# ********************** TESTING CODE ********************** #
# ********************************************************** #

class LSProblemChildTest1el1v(LSProblem):
    """Class for testing a simple problem in 1 variable on 1 element."""
    def __init__(self, theMesh):
        super().__init__(theMesh)
        self.solveLinear()
    def setEquations(self, el):
        opSize = len(self.mesh.GM[el]) / self.mesh.number_of_variables
        opL = {}; opG = {}

        opL['f.f'] = self.mesh.Dx[el].dot(self.mesh.Dx[el])

        opG['f'] = -1.0 * numpy.ones(opSize)

        return opL, opG
    def setBoundaryConditions(self):
        weight = 1.0
        leftValue = 3.0; rightValue = 3.0
        self.Ke[0][0, 0] += weight
        self.Ge[0][0] += weight * leftValue
        self.Ke[-1][-1, -1] += weight
        self.Ge[-1][-1] += weight * rightValue

class LSProblemChildTestNelNv(LSProblem):
    """Class for testing a poisson problem in 2 variables on N elements."""
    def __init__(self, theMesh):
        super().__init__(theMesh)
        self.solveLinear()
    def setEquations(self, el):
        opSize = len(self.mesh.GM[el]) / self.mesh.number_of_variables
        opL = {}; opG = {}

        opL['f.f'] = self.mesh.Dx[el]
        opL['f.g'] = -1.0 * numpy.identity(opSize)
        opL['g.f'] = numpy.zeros((opSize, opSize))
        opL['g.g'] = self.mesh.Dx[el]

        opG['f'] = numpy.zeros(opSize)
        opG['g'] = -1.0 * numpy.ones(opSize)

        return opL, opG
    def setBoundaryConditions(self):
        weight = 1.0
        leftValue = 3.0; rightValue = -1.0
        self.Ke[0][0, 0] += weight
        self.Ge[0][0] += weight * leftValue
        self.Ke[-1][-1, -1] += weight
        self.Ge[-1][-1] += weight * rightValue

class NonLinearProblemTest(LSProblem):
    """Class for testing a poisson problem in 2 variables on N elements."""
    def __init__(self, theMesh):
        super().__init__(theMesh)
        self.solveNonLinear()
    def setEquations(self, el):
        opSize = len(self.mesh.GM[el]) / self.mesh.number_of_variables
        opL = {}; opG = {}
        x = self.mesh.x[self.mesh.GM[el]]

        opL['f.f'] = numpy.diag(self.f[self.mesh.GM[el]]).dot(self.mesh.Dx[el])

        opG['f'] = 2*x**3 - 6*x**2 + 2*x + 2

        return opL, opG
    def setBoundaryConditions(self):
        weight = 1.0
        leftValue = 1.0
        self.Ke[0][0, 0] += weight
        self.Ge[0][0] += weight * leftValue

class TorsionalProblemTest(LSProblem):
    """Class for testing a torsional problem in N variables on N elements."""
    def __init__(self, theMesh):
        super().__init__(theMesh)
        self.solveLinearSlab()
    def setEquations(self, el):
        opL = {}; opG = {}
        opSize = len(self.mesh.GM[el]) / self.mesh.number_of_variables
        x = self.mesh.x[self.mesh.GM[el]]
        Id = numpy.identity(opSize)
        Zero = numpy.zeros(opSize)
        Dx = self.mesh.Dx[el]
        # f = numpy.diag(self.f[self.mesh.GM[el]]) # <--only for non-linear problems

        m=1.0; c=0.2; k=1.0

        opL['v0.v0'] = m*Dx + c*Id
        opL['v0.x0'] = k*Id
        opL['x0.v0'] = Id
        opL['x0.x0'] = -Dx

        opG['v0'] = Zero    #F
        opG['x0'] = Zero    #

        return opL, opG
    def setBoundaryConditions(self):
        weight = 1.0
        initialSpeed = 0.0; initialPosition = 5.0
        self.Ke[0][0, 0] += weight
        self.Ge[0][0] += weight * initialSpeed
        self.Ke[0][5, 5] += weight
        self.Ge[0][5] += weight * initialPosition

class TorsionalProblemTestNv(LSProblem):
    """Class for testing a torsional problem in N variables on N elements."""
    def setEquations(self, el):
        opL = {}; opG = {}
        opSize = len(self.mesh.GM[el]) / self.mesh.number_of_variables
        x = self.mesh.x[self.mesh.GM[el][:opSize]]
        Id = numpy.identity(opSize)
        Zero = numpy.zeros(opSize)
        Dx = self.mesh.Dx[el]
        # f = numpy.diag(self.f[self.mesh.GM[el]]) # <--only for non-linear problems

        m=[2.0, 4.0, 3.0, 10.0]
        c_abs=[1.0, 0.0, 0.0, 0.0]
        c=[0.0, 0.0, 0.0]
        k=[2.0, 7.0, 6.0]


        i = 0
        vi='v0'; vip1='v1'
        xi='x0'; xip1='x1'

        opL[vi +'.'+ vi] = m[i]*Dx + (c[i]+c_abs[i])*Id
        opL[vi +'.'+ xi] = k[i]*Id
        opL[vi +'.'+ vip1] = -1.0*c[i]*Id
        opL[vi +'.'+ xip1] = -1.0*k[i]*Id

        opL[xi +'.'+ vi] = -1.0*Id
        opL[xi +'.'+ xi] = Dx

        opG[vi] = numpy.sin(x/10.0)    #F_1


        n = int(self.mesh.number_of_variables/2 - 1)
        for mass in range(1, n):
            vim1='v'+str(n-1); vi='v'+str(n); vip1='v'+str(n+1)
            xim1='x'+str(n-1); xi='x'+str(n); xip1='x'+str(n+1)

            opL[vi +'.'+ vim1] = -1.0*c[i-1]*Id
            opL[vi +'.'+ xim1] = -1.0*k[i-1]*Id
            opL[vi +'.'+ vi] = m[i]*Dx + (c[i-1]+c[i]+c_abs[i])*Id
            opL[vi +'.'+ xi] = (k[i-1]+k[i])*Id
            opL[vi +'.'+ vip1] = -1.0*c[i]*Id
            opL[vi +'.'+ xip1] = -1.0*k[i]*Id

            opL[xi +'.'+ vi] = -1.0*Id
            opL[xi +'.'+ xi] = Dx


        vim1='v'+str(n-1); vi='v'+str(n)
        xim1='x'+str(n-1); xi='x'+str(n)

        opL[vi +'.'+ vim1] = -1.0*c[n-1]*Id
        opL[vi +'.'+ xim1] = -1.0*k[n-1]*Id
        opL[vi +'.'+ vi] = m[n]*Dx + (c[n-1]+c_abs[n])*Id
        opL[vi +'.'+ xi] = k[n-1]*Id
        opL[xi +'.'+ vi] = -1.0*Id
        opL[xi +'.'+ xi] = Dx

        opG[vi] = Zero    #F_n

        return opL, opG
    def setBoundaryConditions(self):
        initialSpeed = 0.0
        initialPosition = 0.0

        weight = 10.0
        x0index = self.mesh.element_orders[0] + 1

        self.Ke[0][0, 0] += weight
        self.Ge[0][0] += weight * initialSpeed
        self.Ke[0][x0index, x0index] += weight
        self.Ge[0][x0index] += weight * initialPosition


# test_math()
# test_mesh1d()
# testingProblem1el1v()
# testingProblemNelNv()
# testingProblemNonLinear()
# testingProblemTorsional1v()
# testingProblemTorsionalNv()