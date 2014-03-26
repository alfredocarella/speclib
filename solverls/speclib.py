import numpy
import math	# for math.sqrt() in conjGrad()
import matplotlib.pyplot
import os
import pylab


# ********************************************************** #
# ********************** LIBRARY CODE ********************** #
# ********************************************************** #

def conjGradElem(Ke, Ge, GM, dof, x=None, TOL=1.0e-12):
	"""
	Attempts to solve the system of linear equations A*x=b for x. The dof-by-dof coefficient matrix A must be symmetric and positive definite (SPD), and should also be large and sparse. 
	The matrix A is only filled across squared blocks on its main diagonal, as results from a finite-element problem. The list Ke contains the non-empty blocks on matrix A, whose lengths need to match the lengths of the vectors in Ge. The list in GM is the gathering matrix specifying the position of Ke blocks in the matrix A and Ge blocks in vector b.
	               
	Syntax: x = conjGradElem(Ke, Ge, GM, dof, x=None, TOL=1.0e-9)
	<list>	Ke = list of dof-by-dof <numpy.ndarray> matrices (must be SPD)
	<list>	Ge = a list of dof-length <numpy.array> vectors
	<list>	GM = a list of the nodes belonging to each element
	<int>	dof = degrees of freedom of the full system of equations
	<numpy.array>	x = initial iteration value for solution (default is zeros)
	
	Author    : Alfredo R. Carella <alfredocarella@gmail.com>
	"""

	if x == None:
		x = numpy.zeros(dof)
		
	numberOfElements = len(GM)

	# r = loc2gbl(Ge, GM, dof) - loc2gblMatrixVector(Ke, gbl2loc(x, GM), GM, dof)
	r = numpy.zeros(dof)
	for el_ in range(numberOfElements):
		AxLoc = Ke[el_].dot(x[GM[el_]])
		localIndex = numpy.arange(len(GM[el_]))
		r[GM[el_]] += Ge[el_][localIndex] - AxLoc[localIndex]
	
	s = r.copy()
	for i in range(dof):
	
		# u = loc2gblMatrixVector(Ke, gbl2loc(s, GM), GM, dof)
		u = numpy.zeros(dof)
		for el_ in range(numberOfElements):
			Re = Ke[el_].dot(s[GM[el_]])
			localIndex = numpy.arange(len(GM[el_]))
			u[GM[el_]] += Re[localIndex]
				
		alpha = s.dot(r) / s.dot(u)
		x = x + alpha * s
		
		# r = loc2gbl(Ge, GM, dof) - loc2gblMatrixVector(Ke, gbl2loc(x, GM), GM, dof)
		r = numpy.zeros(dof)
		for el_ in range(numberOfElements):
			AxLoc = Ke[el_].dot(x[GM[el_]])
			localIndex = numpy.arange(len(GM[el_]))
			r[GM[el_]] += Ge[el_][localIndex] - AxLoc[localIndex]

		if math.sqrt(r.dot(r)) < TOL:
			break
		else:
			beta = -r.dot(u) / s.dot(u)
			s = r + beta * s
	return x, i		        


def conjGrad(A, b, x=None, TOL=1.0e-12):
	"""
	Attempts to solve the system of linear equations A*x=b for x. The n-by-n coefficient matrix A must be symmetric and positive definite, and should also be large and sparse. The column vector b must have n-length.
	               
	Syntax: x = conjGrad(A, b, x=None, TOL=1.0e-9)
	<numpy.ndarray>	A = n-by-n matrix (must be SPD)
	<numpy.array>	b = n-length vector
	<numpy.array>	x = initial iteration value for solution (default is zeros)
	
	Author    : Alfredo R. Carella <alfredocarella@gmail.com>
	"""
	
	if x == None: 
		x = numpy.zeros(len(b))
		
	n = len(b)
	r = b - A.dot(x)	# Vector product Av(x)
	s = r.copy()
	for i in range(n):
		u = A.dot(s)	# Vector product Av(s)
		alpha = s.dot(r) / s.dot(u)	# Original: np.dot(s,r)/np.dot(s,u)
		x = x + alpha * s
		r = b - A.dot(x)	# Vector product Av(x)
		if math.sqrt(r.dot(r)) < TOL:	# Original: if (math.sqrt(np.dot(r,r))) < TOL:
			break
		else:
			beta = -r.dot(u) / s.dot(u)	# Original: beta = -np.dot(r,u)/np.dot(s,u)
			s = r + beta * s
	return x, i


def info(msg):
	"""
	Prints a message specifying the name of the caller function.
	               
	Syntax: info('str')
	
	Author    : Alfredo R. Carella <alfredocarella@gmail.com>
	"""
	import inspect
	curframe = inspect.currentframe()
	calframe = inspect.getouterframes(curframe, 2)
	print("Msg from \"%s()\": %s" % (calframe[1][3], msg))
	return None


def gaussLegendre(np):
	"""
	Returns separate vectors containing the points and weights for the Gauss Legendre quadrature rule for the interval [-1, 1].
	               
	Syntax: p, w = gaussLobattoLegendre(np)
	<int>	np = number of points
	<float>	p = quadrature points
	<float>	w = quadrature weight
	
	Author    : Alfredo R. Carella <alfredocarella@gmail.com>
	"""

	# This part finds the A-matrix
	A = numpy.zeros((np,np))
	A[0,1] = 1.0
	
	if np > 2:
		for i in range(1,np-1):
			A[i,i-1] = i / (2.0*i + 1.0)
			A[i,i+1] = (i+1) / (2.0*i + 1.0)
	else:
		pass

	A[np-1,np-2] = (np-1.0)/(2.0*np-1.0)
	
	# The array of the sorted eigenvalues/zeros FIXME (probably inefficient)
	eigenValues = numpy.linalg.eigvals(A)
	idx = eigenValues.argsort()   
	p = eigenValues[idx]
	
	# This loop finds the associated weights
	w = numpy.zeros(np)
	for j in range(0,np):
		w[j] = 2.0/((1-p[j]**2.0)*(legendreDerivative(np,p[j]))**2.0)
	return p, w


def gaussLobattoLegendre(np, TOL=1e-18):
	"""
	Returns separate vectors containing the points and weights for the Gauss Lobatto Legendre quadrature rule for the interval [-1, 1].
	               
	Syntax: p, w = gaussLobattoLegendre(np)
	<int>	np = number of points
	<float>	p = quadrature points
	<float>	w = quadrature weight
	
	Author    : Alfredo R. Carella <alfredocarella@gmail.com>
	"""

	TOL = 1e-14	# TOL = Tolerance in the Newton-iteration
	p = numpy.zeros(np)
	p[0] = -1.0
	p[-1] = 1.0
	
	w = numpy.zeros(np)
	
	if np < 3:
		for i in range(np):
			L = legendrePolynomial(np-1, p[i])
			w[i] = 2.0 / ( (np-1) * np * L ** 2.0 )
		return p, w
	else:
		pass
	
	# These points are needed as start (seed) values for the Newton iteration
	GLpoints,GLweights = gaussLegendre(np-1)
	startvalues = numpy.zeros(np) 
	startvalues[1:np-1] = ( GLpoints[0:np-2] + GLpoints[1:np-1] ) / 2.0
	
	# This loop executes the Newton-iteration to find the GLL-points
	for i in range(1,np-1):
		p[i] = startvalues[i]
		p_old = 0.0
		while abs( p[i] - p_old ) > TOL:
			p_old = p[i]
			L = legendrePolynomial(np-1, p_old)
			Ld = legendreDerivative(np-1, p_old)
			p[i] = p_old + ( (1.0 - p_old ** 2.0) * Ld ) / ( (np-1.0) * np * L )

	# This loop finds the associated weights
	for i in range(np):
		L = legendrePolynomial(np-1,p[i])
		w[i] = 2.0 / ( (np-1) * np * L ** 2.0 )

	return p, w


def GLL(np, x_min, x_max):
	"""
	Returns separate vectors containing the points and weights for the Gauss Lobatto Legendre quadrature rule for the interval [x_min, x_max].
	               
	Syntax: p, w = GLL(np, x_min, x_max)
	<int>	np = number of points
	<float>	x_min = left interval boundary
	<float>	x_max = right interval boundary
	<float>	p = quadrature points
	<float>	w = quadrature weight
	
	Author    : Alfredo R. Carella <alfredocarella@gmail.com>
	"""
	
	p, w = gaussLobattoLegendre(np)
	delta = x_max - x_min
	for i in range(np):
		p[i] = delta / 2.0 * (p[i]+1.0) + x_min   # mapping from (-1,1) -> (x_min, x_max)
		w[i] = delta / 2.0 * w[i]
		
	return p, w


def lagrangeDerivativeMatrixGLL(np):
	"""
	Returns a matrix containing the values of the derivatives of the Lagrange polynomials l'_j evaluated at the GLL quadrature points x_i of order np-1, where [-1 <= x_i <= 1]. The obtained matrix (numpy.ndarray) is defined as: D_{ij} = l'_j(x_i) for i,j = 0:np-1
	               
	Syntax: D = lagrangeDerivativeMatrixGLL(np,x)
	<int>	np = number of points
	
	Author    : Alfredo R. Carella <alfredocarella@gmail.com>
	"""

	D = numpy.zeros((np,np))
	GLLPoints,GLLWeights = gaussLobattoLegendre(np)
	
	for i in range(np):
		for j in range(np):
		
			if i==j:
				pass	# D[i,j]=0 for the main diagonal
			else:
				D[i,j] = legendrePolynomial(np-1, GLLPoints[i]) / (legendrePolynomial(np-1, GLLPoints[j]) * (GLLPoints[i] - GLLPoints[j])) #Eq. 4.34 in DeMaerschalck2003
			
	D[0,0] = -np * (np-1) / 4.0
	D[np-1, np-1] = np * (np-1) / 4.0	
	
	return D

def lagrangeInterpolantMatrix(xIn, xOut):
	"""
	Returns a matrix 'L' that yields 'f(x_out)=L*f(x_in)', where x_in are the gauss-lobatto-legendre interpolating nodes of order n+1 and x_out is an arbitrary set of points.
	"""
	
	npIn = len(xIn)
	npOut = len(xOut)
	L = numpy.ones((npOut, npIn))
	# % Sub-index i_basis goes with the interpolant basis
	# % Sub-index k_coord goes with (x) the evaluation coordinate
	for i_basis in range(npIn):	# interpolant basis
		for j_basis in range(npIn):	# evaluation coordinates
			if i_basis!=j_basis: 
				for k_coord in range(npOut):
					L[k_coord, i_basis] *= (xOut[k_coord]-xIn[j_basis]) / (xIn[i_basis]-xIn[j_basis])
	
	return L	# Interpolant matrix


def legendreDerivative(n, x):
	"""
	Returns the the value of the derivative of the n'th Legendre polynomial evaluated at the coordinate x. The input 'x' can be a vector of points.

	Syntax: Ld = legendreDerivative(n,x)
	<int>	n = polynomial order 0,1,...
	<float>	x = coordinate -1 =< x =< 1
	
	Author    : Alfredo R. Carella <alfredocarella@gmail.com>
	"""	

	Ln = numpy.zeros(n+1)
	Ln[0] = 1.0
	Ln[1] = x
	
	if x == -1  or  x == 1:
		Ld = x**(n-1.0) * (1.0/2.0) * n * (n+1.0)
	else:
		for i in range(1,n):
			Ln[i+1] = (2.0*i+1)/(i+1.0)*x*Ln[i] - i/(i+1.0)*Ln[i-1] #Recurrence 4.5.10 
		Ld = n/(1.0-x**2.0)*Ln[n-1] - n*x/(1.0-x**2.0)*Ln[n]
	
	return Ld

def legendrePolynomial(n, x):
	"""
	Returns the value of the n'th Legendre polynomial evaluated at the coordinate 'x'. The input 'x' can be a vector of points.

	Syntax: L = legendrePolynomial(n,x)
	<int>	n = polynomial order 0,1,...
	<float>	x = coordinate -1 =< x =< 1
	
	Author    : Alfredo R. Carella <alfredocarella@gmail.com>
	"""	
	
	Ln = numpy.zeros(( n+1, len(numpy.atleast_1d(x)) ))
	Ln[0,:] = 1.0
	Ln[1,:] = x
	
	if n > 1:
		for i in range(1, n):
			Ln[i+1, :] = ( (2.0 * i+1.0) / (i+1.0) * x) * Ln[i,:] - i / (i+1.0) * Ln[i-1, :] # Recurrence 4.5.10 in 'Press1993.pdf'
	else:
		pass
		
	return Ln[n,:]


class Iterator(object):

	def __init__(self, MIN_RESIDUAL=1e-20, MAX_NONLINEAR_ITERS=50, MIN_DELTA=1e-16):
		self.MIN_RESIDUAL = MIN_RESIDUAL
		self.MAX_NONLINEAR_ITERS = MAX_NONLINEAR_ITERS
		self.MIN_DELTA = MIN_DELTA
		
		self.delta = 1.0
		self.numIter = 0
		self.converged = False
		self.not_converging = False
		self.reached_max_iters = False	

	def iteratePicard(self, theProblem, setOperators, setBoundaryConditions, listOfElements=None):
		while (not self.converged) and (not self.not_converging) and (not self.reached_max_iters):
		
			setOperators(listOfElements)
			setBoundaryConditions()
			theProblem.f_old = theProblem.f.copy()

			if len(theProblem.mesh.GM[0]) == theProblem.mesh.dofNv:
				theProblem.f, numItersCG = conjGrad(theProblem.Ke[0], theProblem.Ge[0])
			else:
				theProblem.f, numItersCG = conjGradElem(theProblem.Ke, theProblem.Ge, theProblem.mesh.GM, theProblem.mesh.dofNv)

			theProblem.residual = theProblem.computeResidual(listOfElements)
			self.delta = numpy.linalg.norm(theProblem.f - theProblem.f_old) / numpy.linalg.norm(theProblem.f)

			self.numIter += 1
			
			if (theProblem.residual < self.MIN_RESIDUAL):
				# print("Converged: residual below tolerance. Residual < %r" % it.MIN_DELTA)
				self.converged = True
			elif (self.delta < self.MIN_DELTA):
				print("Equal consecutive nonlinear iterations. Delta = %r" % self.delta)
				self.not_converging = True
			elif (self.numIter >= self.MAX_NONLINEAR_ITERS):
				print("Stopping after having reached %r nonlinear iterations." % self.numIter)
				self.reached_max_iters = True
			else:
				pass


class Mesh1d(object):
	"""
	This class represents a 1-dimensional high-order Finite Element mesh. 
	macroGrid = vector with the position of element boundary nodes
	P = vector with the order for each element (int)
	NV = number of variables (int)
	
	The attributes of class Mesh1d()are:
	
	Mesh1d.macroNodes
	Mesh1d.elementOrders
	Mesh1d.numberOfElements
	Mesh1d.listOfVariables
	Mesh1d.numberOfVariables
	Mesh1d.dof1v
	Mesh1d.dofNv
	Mesh1d.GM
	Mesh1d.GM1v
	Mesh1d.quadWeights
	Mesh1d.quadPoints
	Mesh1d.Jx
	Mesh1d.Dx
	Mesh1d.x
	Mesh1d.longQuadWeights
		
	Example:
	macroGrid, P, NV = numpy.array((0.0,1.0,2.0,3.0)), numpy.array((3,4,2)), 2
	myMesh1d = Mesh1d(macroGrid, P, NV)
	print("myMesh1d.Dx[0] = \n%r" % myMesh1d.Dx[0])
	print("myMesh1d.x[myMesh1d.GM1v[0]] = %r" % myMesh1d.x[myMesh1d.GM1v[0]])
	print("myMesh1d.Dx[0].dot(myMesh1d.quadPoints[0]) = \n%r\" % myMesh1d.Dx[0].dot(myMesh1d.quadPoints[0]))
	"""
	
	def __init__(self, macroGrid, P, varNames=['f']):
		self.macroNodes = macroGrid
		self.elementOrders = numpy.atleast_1d(P)
		self.numberOfElements = len(self.elementOrders)
		self.listOfVariables = varNames
		self.numberOfVariables = len(self.listOfVariables)

		self.setGM1d()
		
		self.quadWeights = []
		self.quadPoints = []
		self.Jx = []
		self.Dx = []
		self.x = numpy.zeros(self.dof1v)
		self.longQuadWeights = []
		self.pos = []
		for el_ in range(self.numberOfElements):
			lowElemBoundary = self.macroNodes[el_]
			uppElemBoundary = self.macroNodes[el_+1]
			self.Jx.append( (uppElemBoundary - lowElemBoundary) / 2.0 )
			self.Dx.append( lagrangeDerivativeMatrixGLL(self.elementOrders[el_]+1) / self.Jx[el_] )
			qx, qw = GLL(self.elementOrders[el_]+1, lowElemBoundary, uppElemBoundary)
			self.quadPoints.append(qx)	# x coordinate (quad points)
			self.quadWeights.append(qw)	# quadrature weights
			longQw = numpy.tile(qw, self.numberOfVariables)	
			self.longQuadWeights.append(longQw)
			self.pos.append({})
			for var_ in range(self.numberOfVariables):
				firstNodeInVar = var_ * len(self.GM1v[el_])
				lastNodeInVar = firstNodeInVar + len(self.GM1v[el_])
				self.pos[el_][self.listOfVariables[var_]] = numpy.arange(firstNodeInVar, lastNodeInVar)
			self.x[self.GM1v[el_]] = qx
			
		self.x = numpy.tile(self.x, self.numberOfVariables)

	def plotMesh(self):
		# Plot nodes and line
		xMicro = self.x
		xMacro = self.macroNodes
		matplotlib.pyplot.plot((xMacro[0], xMacro[-1]), (0,0), 'r--', linewidth=2.0)	# Lines
		matplotlib.pyplot.plot(xMicro, xMicro*0, 'ro')	# Nodes (micro)
		matplotlib.pyplot.plot(xMacro, xMacro*0, 'bs', markersize=10)	# Nodes (macro)
		
		# Plot node and element numbers
		for node_ in range(self.dof1v):
			matplotlib.pyplot.text(self.x[node_], -0.1, str(node_), fontsize=10, color='red')
		for border_ in range(len(xMacro)-1):
			elemCenter = ( xMacro[border_] + xMacro[border_+1] ) / 2.0
			matplotlib.pyplot.text(elemCenter, +0.1, str(border_), fontsize=15, color='blue')

		# Write annotations
		firstElementCenter = ( xMacro[0] + xMacro[1] ) / 2.0
		matplotlib.pyplot.annotate('element numbers', xy=(firstElementCenter, 0.17), xytext=(firstElementCenter, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
		matplotlib.pyplot.annotate('node number', xy=(xMicro[1], -0.12), xytext=(xMicro[1], -0.3), arrowprops=dict(facecolor='black', shrink=0.05))
		matplotlib.pyplot.text((xMacro[0]+xMacro[-1])/4.0, -0.9, 'Degrees of freedom per variable: %d\nTotal degrees of freedom: %d\nNumber of elements: %d\nVariables: %d %s' %(self.dof1v, self.dofNv, self.numberOfElements, self.numberOfVariables, self.listOfVariables))
		matplotlib.pyplot.title('1-D mesh information')
		matplotlib.pyplot.xlabel("Independent variable coordinate")
		matplotlib.pyplot.axis([xMacro[0], xMacro[-1], -1, 1])
		matplotlib.pyplot.show()

	def setGM1d(self):
		self.dof1v = numpy.sum(self.elementOrders)+1
		self.dofNv = self.dof1v * self.numberOfVariables
		
		nodeCnt1v_ = 0
		self.GM = []
		self.GM1v = []
		for el_ in range(self.numberOfElements):
			elementSize = self.numberOfVariables * (self.elementOrders[el_]+1)
			self.GM.append(numpy.zeros(elementSize, dtype=numpy.int))
			for var_ in range(self.numberOfVariables):
				firstVarPos = var_ * (self.elementOrders[el_]+1)
				lastVarPos = firstVarPos + self.elementOrders[el_]
				firstVarNumber = var_ * self.dof1v + nodeCnt1v_
				lastVarNumber = firstVarNumber + self.elementOrders[el_]				
				self.GM[el_][firstVarPos : lastVarPos + 1] = numpy.arange(firstVarNumber, lastVarNumber + 1)
			nodeCnt1v_ += self.elementOrders[el_]
			self.GM1v.append( self.GM[el_][0 : self.elementOrders[el_]+1] )


class LSProblem(object):

	def __init__(self, theMesh):
		self.mesh = theMesh

	def computeResidual(self, listOfElements=None):
		"""Compute the Least-Squares total residual."""
		
		if listOfElements == None: 	listOfElements = range(len(self.mesh.GM))
		
		residual = 0.0
		for el_ in listOfElements:
			W = self.mesh.longQuadWeights[el_]
			GM = self.mesh.GM[el_]
			opG = self.opG[el_]
			opL = self.opL[el_]
			residual += W.dot((opL.dot(self.f[GM])-opG)**2)	#Int[(Lf-G)**2] + [f_a-f(a)]**2   <---The second term is missing. Therefore this formulation does not currently consider compliance with boundary conditions!
		return residual

	def plotSolution(self, varList=None, filename=None):
		info("This routine still needs to be polished and documented.")
		
		if varList == None: varList = self.mesh.listOfVariables
		
		fig = matplotlib.pyplot.figure()
		for varName in varList:
			varNumber = self.mesh.listOfVariables.index(varName)
			xIn = numpy.zeros([])
			yIn = numpy.zeros([])
			xOut = numpy.zeros([])
			yOut = numpy.zeros([]) 
			for elem in range(self.mesh.numberOfElements):
				varIndices = self.mesh.GM[elem][self.mesh.pos[elem][varName]]
				xInElem = self.mesh.x[varIndices]
				yInElem = self.f[varIndices]
				xOutElem = numpy.linspace(self.mesh.macroNodes[elem], self.mesh.macroNodes[elem+1], 10)
				yOutElem = lagrangeInterpolantMatrix(xInElem, xOutElem).dot(yInElem)
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
		info("THIS FUNCTION MUST BE OVERRIDEN IN CHILD CLASS")

	def setSlabBoundaryConditions(self, elem):
		weight = 1.0
		for varName in self.mesh.listOfVariables:
		
			finalValueIndices = self.mesh.GM[elem-1][self.mesh.pos[elem-1][varName]]
			initialValueIndices = self.mesh.pos[elem][varName]
			f_index = finalValueIndices[-1]
			gk_index = initialValueIndices[0]
			
			self.Ke[0][gk_index, gk_index] += weight
			self.Ge[0][gk_index] += weight * self.f[f_index]

	def setEquations(self, el):
		info("THIS FUNCTION MUST BE OVERRIDEN IN CHILD CLASS")

	def setOperators(self, listOfElements=None):
	
		if listOfElements == None:
			listOfElements = range(self.mesh.numberOfElements)
		else:
			listOfElements = list(listOfElements)
			
		self.opL = []
		self.opG = []
		self.Ke = []
		self.Ge = []

		# Create the operators
		for elem in listOfElements:
			el_ = listOfElements.index(elem)
			elemSize = len(self.mesh.GM[elem])
			self.opL.append( numpy.zeros((elemSize, elemSize)) )
			self.opG.append( numpy.zeros((elemSize)) )
			
			opL_dict, opG_dict = self.setEquations(elem)

			for varRow_ in self.mesh.listOfVariables:
				for varCol_ in self.mesh.listOfVariables:
					if (varRow_+'.'+varCol_) in opL_dict:
						self.opL[el_][numpy.ix_(self.mesh.pos[elem][varRow_], self.mesh.pos[elem][varCol_])] += opL_dict[varRow_+'.'+varCol_]
				if varRow_ in opG_dict:
					self.opG[el_][self.mesh.pos[elem][varRow_]] += opG_dict[varRow_]

			# Generate problem sub-matrices
			elemNodes = self.mesh.GM[elem]
			LW = self.opL[el_].T.dot(numpy.diag(self.mesh.longQuadWeights[elem]))
			self.Ke.append( LW.dot(self.opL[el_]) )
			self.Ge.append( LW.dot(self.opG[el_]) )	

	def setSolution(self, f):
		self.fOld = f

	def solveLinear(self):
		# it = Iterator(MIN_RESIDUAL=1e-20, MAX_NONLINEAR_ITERS=50, MIN_DELTA=1e-16)
		# self.f = self.mesh.x	# or even --> numpy.ones(len(self.mesh.x))
		# it.iteratePicard(self, self.setOperators, self.setBoundaryConditions)
		self.setOperators()
		self.setBoundaryConditions()
		self.f, numIters = conjGradElem(self.Ke, self.Ge, self.mesh.GM, self.mesh.dofNv)

	def solveNonLinear(self):
		self.f = self.mesh.x	# or even --> numpy.ones(len(self.mesh.x))
		it = Iterator(MIN_RESIDUAL=1e-20, MAX_NONLINEAR_ITERS=50, MIN_DELTA=1e-16)
		it.iteratePicard(self, self.setOperators, self.setBoundaryConditions, [0,1])

		print("Iterations: %r  -  Residual: %04.2e  -  delta = %04.2e" % (it.numIter, self.residual, it.delta))

	def solveLinearSlab(self):
		self.f = numpy.zeros(self.mesh.dofNv)
	
		self.setOperators([0])
		self.setBoundaryConditions()
		f_elem, numIters = conjGrad(self.Ke[0], self.Ge[0])
		self.f[self.mesh.GM[0]] = f_elem
		
		for el_ in range(1,self.mesh.numberOfElements):
			self.setOperators([el_])
			self.setSlabBoundaryConditions(el_)
			f_elem, numIters = conjGrad(self.Ke[0], self.Ge[0])
			self.f[self.mesh.GM[el_]] = f_elem

	def solveNonLinearSlab(self):
		info("Would be very useful, but it is complex to implement right now.")
