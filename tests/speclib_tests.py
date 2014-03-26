from nose.tools import *
from solverls import speclib

import numpy


def test_math():	# Testing some of the mathematical routines
	"""
	# The dependency-tree 'speclib' is:
	#
	# *GLL(np, x_min, x_max)
	# 	*gaussLobattoLegendre(np)
	# 		*gaussLegendre(np)
	# 			*legendreDerivative(n,x)
	# 		*legendreDerivative(n, x)
	# 		*legendrePolynomial(n, x)
	# *lagrangeDerivativeMatrixGLL(np)
	# 	*gaussLobattoLegendre(np)
	# 	*legendrePolynomial(n, x)
	# *lagrangeInterpolantMatrix(xIn, xOut)
	#
	# I automatically test only the higher level functions, assuming that if they are OK, then the functions used by them are also OK
	"""
	
	np=5
	x_min, x_max = 0, 4
	nInterp = 30
	delta = x_max - x_min
	
	# speclib.GLL(np, x_min, x_max) tested here
	p, w = speclib.GLL(np, x_min, x_max)
	assert_equal(p[0], x_min)
	assert_equal(p[-1], x_max)
	assert_almost_equal(numpy.sum(w), (delta))
	
	# speclib.lagrangeDerivativeMatrixGLL(np) tested here	
	Dx = speclib.lagrangeDerivativeMatrixGLL(np) * 2.0/delta
	Dp = Dx.dot(p)
	numpy.testing.assert_allclose(Dp, numpy.ones(np))

	# speclib.lagrangeInterpolantMatrix(x_in, x_out) tested here	
	x_in = p
	x_out = numpy.linspace(x_min, x_max, nInterp)
	L = speclib.lagrangeInterpolantMatrix(x_in, x_out)
	numpy.testing.assert_allclose(L.dot(Dp), numpy.ones(nInterp))
	
	speclib.info("Execution complete!")
	

def test_Mesh1d():	# Testing the mesh generation (plotting is not tested here)
	macroGrid, P, varList = numpy.array((0.0,1.0,2.0,3.0)), numpy.array((3,4,2)), ['T', 'pres', 'quality']	
	myMesh1d = speclib.Mesh1d(macroGrid, P, varList)
	
	numpy.testing.assert_array_equal(myMesh1d.elementOrders, P)
	numpy.testing.assert_allclose(myMesh1d.macroNodes, macroGrid)
	assert_equal(myMesh1d.listOfVariables, varList)
	assert_equal(myMesh1d.numberOfElements, len(myMesh1d.elementOrders))
	assert_equal(myMesh1d.numberOfVariables, len(myMesh1d.listOfVariables))
	assert_equal(myMesh1d.dof1v, sum(myMesh1d.elementOrders)+1)
	assert_equal(myMesh1d.dofNv, myMesh1d.numberOfVariables*myMesh1d.dof1v)
	assert_equal(len(myMesh1d.GM), myMesh1d.numberOfElements)
	assert_equal(len(myMesh1d.GM1v), myMesh1d.numberOfElements)

	integralTestValue = 0
	for el in range(myMesh1d.numberOfElements):
		integralTestValue += myMesh1d.quadPoints[el].dot(myMesh1d.quadWeights[el])
		assert_almost_equal(myMesh1d.Jx[el], (myMesh1d.x[myMesh1d.GM1v[el][-1]]-myMesh1d.x[myMesh1d.GM1v[el][0]])/2.0 )
		numpy.testing.assert_array_equal(myMesh1d.GM[el][:myMesh1d.elementOrders[el]+1], myMesh1d.GM1v[el])
		numpy.testing.assert_array_equal(myMesh1d.quadPoints[el], myMesh1d.x[myMesh1d.GM1v[el]])
		numpy.testing.assert_array_equal(myMesh1d.Dx[el], speclib.lagrangeDerivativeMatrixGLL(myMesh1d.elementOrders[el]+1)/myMesh1d.Jx[el])
		numpy.testing.assert_array_equal(myMesh1d.longQuadWeights[el], numpy.tile(myMesh1d.quadWeights[el], myMesh1d.numberOfVariables))
		posVarTestValue = []
		for var in myMesh1d.listOfVariables:
			posVarTestValue = numpy.append(posVarTestValue, myMesh1d.pos[el][var])
		numpy.testing.assert_array_equal(posVarTestValue, range(len(myMesh1d.GM[el])))
	assert_almost_equal(integralTestValue, (myMesh1d.macroNodes[-1]-myMesh1d.macroNodes[0])**2 /2)
		
	
def testingProblem1el1v():	# Testing results for a simple problem (1 var, 1 elem)
	macroGrid, P, varList = numpy.array((0.0,2.0)), numpy.array((4)), ['f']
	myMesh1d = speclib.Mesh1d(macroGrid, P, varList)

	myProblem = speclib.LSProblemChildTest1el1v(myMesh1d)

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

def testingProblemNelNv():	# Testing a problem w/ multiple variables and elements
	macroGrid, P, varList = numpy.array((0.0, 1.0, 2.0)), numpy.array((3, 3)), ['f', 'g']
	print("macroGrid = %r - P = %r - varList = %r" % (macroGrid, P, varList))
	myMesh1d = speclib.Mesh1d(macroGrid, P, varList)
	print("myMesh1d = Mesh1d(macroGrid, P, varList)")

	myProblem = speclib.LSProblemChildTestNelNv(myMesh1d)
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

def testingProblemNonLinear():	# Testing iterative routine for solving a non-linear problem
	macroGrid, P, varList = numpy.array((0.0, 1.0, 2.0)), numpy.array((3, 3)), ['f']
	print("macroGrid = %r - P = %r - varList = %r" % (macroGrid, P, varList))

	myMesh1d = speclib.Mesh1d(macroGrid, P, varList)
	myProblem = speclib.NonLinearProblemTest(myMesh1d)
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

def testingProblemTorsional1v():	# Testing a torsional vibration problem (1 mass)
	macroGrid = numpy.linspace(0.0, 30.0, 50)
	P = [4] * (len(macroGrid)-1)
	varList = ['v0', 'x0']
	# print("macroGrid = %r - P = %r - varList = %r" % (macroGrid, P, varList))

	myMesh1d = speclib.Mesh1d(macroGrid, P, varList)
	myProblem = speclib.TorsionalProblemTest(myMesh1d)
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

def testingProblemTorsionalNv():	# Testing a torsional vibration problem (N masses)
	macroGrid = numpy.linspace(0.0, 30.0, 40)
	P = [5] * (len(macroGrid)-1)
	numberOfMasses = 2

	varList = []
	for varNum in range(numberOfMasses):
		varList.append('v%d' % varNum)
		varList.append('x%d' % varNum)
	print(varList)

	myMesh1d = speclib.Mesh1d(macroGrid, P, varList)
	myProblem = speclib.TorsionalProblemTestNv(myMesh1d)
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
