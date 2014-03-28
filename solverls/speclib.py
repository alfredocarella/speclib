import math    # for math.sqrt() in conjGrad()

import numpy
import matplotlib.pyplot


# ********************************************************** #
# ********************** LIBRARY CODE ********************** #
# ********************************************************** #

def conj_grad_elem(k_elem, Ge, GM, dof, x=None, TOL=1.0e-12):
    """
    Attempts to solve the system of linear equations A*x=b for x. The dof-by-dof coefficient matrix A must be symmetric and positive definite (SPD), and should also be large and sparse. 
    The matrix A is only filled across squared blocks on its main diagonal, as results from a finite-element problem. The list Ke contains the non-empty blocks on matrix A, whose lengths need to match the lengths of the vectors in Ge. The list in GM is the gathering matrix specifying the position of Ke blocks in the matrix A and Ge blocks in vector b.
                   
    Syntax: x = conjGradElem(Ke, Ge, GM, dof, x=None, TOL=1.0e-9)
    <list>    Ke = list of dof-by-dof <numpy.ndarray> matrices (must be SPD)
    <list>    Ge = a list of dof-length <numpy.array> vectors
    <list>    GM = a list of the nodes belonging to each element
    <int>    dof = degrees of freedom of the full system of equations
    <numpy.array>    x = initial iteration value for solution (default is zeros)
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """

    if x is None:
        x = numpy.zeros(dof)

    number_of_elements = len(GM)

    # r = loc2gbl(Ge, GM, dof) - loc2gblMatrixVector(Ke, gbl2loc(x, GM), GM, dof)
    r = numpy.zeros(dof)
    for el_ in range(number_of_elements):
        mat_vec_local_product = k_elem[el_].dot(x[GM[el_]])
        local_index = numpy.arange(len(GM[el_]))
        r[GM[el_]] += Ge[el_][local_index] - mat_vec_local_product[local_index]

    s = r.copy()
    for i in range(dof):

        # u = loc2gblMatrixVector(Ke, gbl2loc(s, GM), GM, dof)
        u = numpy.zeros(dof)
        for el_ in range(number_of_elements):
            r_elem = k_elem[el_].dot(s[GM[el_]])
            local_index = numpy.arange(len(GM[el_]))
            u[GM[el_]] += r_elem[local_index]

        alpha = s.dot(r) / s.dot(u)
        x = x + alpha * s

        # r = loc2gbl(Ge, GM, dof) - loc2gblMatrixVector(Ke, gbl2loc(x, GM), GM, dof)
        r = numpy.zeros(dof)
        for el_ in range(number_of_elements):
            mat_vec_local_product = k_elem[el_].dot(x[GM[el_]])
            local_index = numpy.arange(len(GM[el_]))
            r[GM[el_]] += Ge[el_][local_index] - mat_vec_local_product[local_index]

        if math.sqrt(r.dot(r)) < TOL:
            break
        else:
            beta = -r.dot(u) / s.dot(u)
            s = r + beta * s
    return x, i


def conj_grad(a, b, x=None, tol=1.0e-12):
    """
    Attempts to solve the system of linear equations A*x=b for x. The n-by-n coefficient matrix A must be symmetric and positive definite, and should also be large and sparse. The column vector b must have n-length.
                   
    Syntax: x = conjGrad(a, b, x=None, TOL=1.0e-9)
    <numpy.ndarray>    a = n-by-n matrix (must be SPD)
    <numpy.array>    b = n-length vector
    <numpy.array>    x = initial iteration value for solution (default is zeros)
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """

    if x is None:
        x = numpy.zeros(len(b))

    n = len(b)
    r = b - a.dot(x)    # Vector product Av(x)
    s = r.copy()
    for i in range(n):
        u = a.dot(s)    # Vector product Av(s)
        alpha = s.dot(r) / s.dot(u)    # Original: np.dot(s,r)/np.dot(s,u)
        x = x + alpha * s
        r = b - a.dot(x)    # Vector product Av(x)
        if math.sqrt(r.dot(r)) < tol:    # Original: if (math.sqrt(np.dot(r,r))) < TOL:
            break
        else:
            beta = -r.dot(u) / s.dot(u)    # Original: beta = -np.dot(r,u)/np.dot(s,u)
            s = r + beta * s
    return x, i


def info(msg):
    """
    Prints a message specifying the name of the caller function.
                   
    Syntax: info('str')
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """
    import inspect
    current_frame = inspect.currentframe()
    call_frame = inspect.getouterframes(current_frame, 2)
    print("Msg from \"%s()\": %s" % (call_frame[1][3], msg))
    return None


def gauss_legendre(np):
    """
    Returns separate vectors containing the points and weights for the Gauss Legendre quadrature rule for the interval [-1, 1].
                   
    Syntax: p, w = gauss_lobatto_legendre(np)
    <int>    np = number of points
    <float>    p = quadrature points
    <float>    w = quadrature weight
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """

    # This part finds the A-matrix
    a = numpy.zeros((np,np))
    a[0, 1] = 1.0

    if np > 2:
        for i in range(1, np-1):
            a[i, i-1] = i / (2.0*i + 1.0)
            a[i, i+1] = (i+1) / (2.0*i + 1.0)
    else:
        pass

    a[np-1, np-2] = (np-1.0)/(2.0*np-1.0)

    # The array of the sorted eigenvalues/zeros FIXME (probably inefficient)
    eigenvalues = numpy.linalg.eigvals(a)
    idx = eigenvalues.argsort()
    p = eigenvalues[idx]

    # This loop finds the associated weights
    w = numpy.zeros(np)
    for j in range(0,np):
        w[j] = 2.0/((1-p[j]**2.0)*(legendre_derivative(np,p[j]))**2.0)
    return p, w


def gauss_lobatto_legendre(np, tol=1e-18):  # TODO: gauss_lobatto_legendre() should be inside gll(), as a particular case

    """
    Returns separate vectors containing the points and weights for the Gauss Lobatto Legendre quadrature rule for the interval [-1, 1].
                   
    Syntax: p, w = gauss_lobatto_legendre(np)
    <int>    np = number of points
    <float>    p = quadrature points
    <float>    w = quadrature weight
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """

    tol = 1e-14    # TOL = Tolerance in the Newton-iteration
    p = numpy.zeros(np)
    p[0] = -1.0
    p[-1] = 1.0

    w = numpy.zeros(np)

    if np < 3:
        for i in range(np):
            L = legendre_polynomial(np-1, p[i])
            w[i] = 2.0 / ( (np-1) * np * L ** 2.0 )
        return p, w
    else:
        pass

    # These points are needed as start (seed) values for the Newton iteration
    gl_points, gl_weights = gauss_legendre(np-1)
    start_values = numpy.zeros(np)
    start_values[1:np-1] = (gl_points[0:np-2] + gl_points[1:np-1]) / 2.0

    # This loop executes the Newton-iteration to find the GLL-points
    for i in range(1, np-1):
        p[i] = start_values[i]
        p_old = 0.0
        while abs(p[i] - p_old) > tol:
            p_old = p[i]
            L = legendre_polynomial(np-1, p_old)
            Ld = legendre_derivative(np-1, p_old)
            p[i] = p_old + ((1.0 - p_old ** 2.0) * Ld) / ((np-1.0) * np * L)

    # This loop finds the associated weights
    for i in range(np):
        L = legendre_polynomial(np-1,p[i])
        w[i] = 2.0 / ( (np-1) * np * L ** 2.0 )

    return p, w


def gll(np, x_min, x_max):
    """
    Returns separate vectors containing the points and weights for the Gauss Lobatto Legendre quadrature rule for the interval [x_min, x_max].
                   
    Syntax: p, w = gll(np, x_min, x_max)
    <int>    np = number of points
    <float>    x_min = left interval boundary
    <float>    x_max = right interval boundary
    <float>    p = quadrature points
    <float>    w = quadrature weight
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """

    p, w = gauss_lobatto_legendre(np)
    delta = x_max - x_min
    for i in range(np):
        p[i] = delta / 2.0 * (p[i]+1.0) + x_min   # mapping from (-1,1) -> (x_min, x_max)
        w[i] *= delta / 2.0

    return p, w


def lagrange_derivative_matrix_gll(np):
    """
    Returns a matrix containing the values of the derivatives of the Lagrange polynomials l'_j evaluated at the GLL quadrature points x_i of order np-1, where [-1 <= x_i <= 1]. The obtained matrix (numpy.ndarray) is defined as: D_{ij} = l'_j(x_i) for i,j = 0:np-1
                   
    Syntax: D = lagrange_derivative_matrix_gll(np, x)
    <int>    np = number of points
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """

    derivative_matrix = numpy.zeros((np, np))
    gll_points, gll_weights = gauss_lobatto_legendre(np)

    for i in range(np):
        for j in range(np):

            if i == j:
                pass    # D[i,j]=0 for the main diagonal
            else:
                derivative_matrix[i, j] = legendre_polynomial(np-1, gll_points[i]) / (legendre_polynomial(np-1, gll_points[j]) * (gll_points[i] - gll_points[j]))  # Eq. 4.34 in DeMaerschalck2003

    derivative_matrix[0, 0] = -np * (np-1) / 4.0
    derivative_matrix[np-1, np-1] = np * (np-1) / 4.0

    return derivative_matrix


def lagrange_interpolating_matrix(x_in, x_out):
    """
    Returns a matrix 'L' that yields 'f(x_out)=L*f(x_in)', where x_in are the gauss-lobatto-legendre interpolating nodes of order n+1 and x_out is an arbitrary set of points.
    """

    input_length = len(x_in)
    output_length = len(x_out)
    interpolating_matrix = numpy.ones((output_length, input_length))
    # % Sub-index i_basis goes with the interpolating basis
    # % Sub-index k_coord goes with (x) the evaluation coordinate
    for i_basis in range(input_length):    # interpolating basis
        for j_basis in range(input_length):    # evaluation coordinates
            if i_basis != j_basis:
                for k_coord in range(output_length):
                    interpolating_matrix[k_coord, i_basis] *= (x_out[k_coord]-x_in[j_basis]) / (x_in[i_basis]-x_in[j_basis])

    return interpolating_matrix    # Interpolating matrix


def legendre_derivative(n, x):
    """
    Returns the the value of the derivative of the n'th Legendre polynomial evaluated at the coordinate x. The input 'x' can be a vector of points.

    Syntax: Ld = legendre_derivative(n,x)
    <int>    n = polynomial order 0,1,...
    <float>    x = coordinate -1 =< x =< 1
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """

    lagrange_polynomial = numpy.zeros(n+1)
    lagrange_polynomial[0] = 1.0
    lagrange_polynomial[1] = x

    if x == -1 or x == 1:
        lagrange_derivative = x**(n-1.0) * (1.0/2.0) * n * (n+1.0)
    else:
        for i in range(1,n):
            lagrange_polynomial[i+1] = (2.0*i+1)/(i+1.0)*x*lagrange_polynomial[i] - i/(i+1.0)*lagrange_polynomial[i-1]  # Recurrence 4.5.10
        lagrange_derivative = n/(1.0-x**2.0)*lagrange_polynomial[n-1] - n*x/(1.0-x**2.0)*lagrange_polynomial[n]

    return lagrange_derivative


def legendre_polynomial(n, x):
    """
    Returns the value of the n'th Legendre polynomial evaluated at the coordinate 'x'. The input 'x' can be a vector of points.

    Syntax: L = legendre_polynomial(n,x)
    <int>    n = polynomial order 0,1,...
    <float>    x = coordinate -1 =< x =< 1
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """

    Ln = numpy.zeros((n+1, len(numpy.atleast_1d(x))))
    Ln[0, :] = 1.0
    Ln[1, :] = x

    if n > 1:
        for i in range(1, n):
            Ln[i+1, :] = ( (2.0 * i+1.0) / (i+1.0) * x) * Ln[i,:] - i / (i+1.0) * Ln[i-1, :]  # Recurrence 4.5.10 in 'Press1993.pdf'
    else:
        pass

    return Ln[n, :]


class Iterator(object):

    def __init__(self, min_residual=1e-20, max_nonlinear_it=50, min_delta=1e-16):
        self.min_residual = min_residual
        self.max_nonlinear_it = max_nonlinear_it
        self.min_delta = min_delta

        self.delta = 1.0
        self.number_of_iterations = 0
        self.converged = False
        self.not_converging = False
        self.reached_max_it = False

    def iteratePicard(self, problem, set_operators, set_boundary_conditions, list_of_elements=None):
        while (not self.converged) and (not self.not_converging) and (not self.reached_max_it):

            set_operators(list_of_elements)
            set_boundary_conditions()
            problem.f_old = problem.f.copy()

            if len(problem.mesh.GM[0]) == problem.mesh.dofNv:
                problem.f, num_cg_it = conj_grad(problem.Ke[0], problem.Ge[0])
            else:
                problem.f, num_cg_it = conj_grad_elem(problem.Ke, problem.Ge, problem.mesh.GM, problem.mesh.dofNv)

            problem.residual = problem.computeResidual(list_of_elements)
            self.delta = numpy.linalg.norm(problem.f - problem.f_old) / numpy.linalg.norm(problem.f)

            self.number_of_iterations += 1

            if problem.residual < self.min_residual:
                # print("Converged: residual below tolerance. Residual < %r" % it.MIN_DELTA)
                self.converged = True
            elif self.delta < self.min_delta:
                print("Equal consecutive nonlinear iterations. Delta = %r" % self.delta)
                self.not_converging = True
            elif self.number_of_iterations >= self.max_nonlinear_it:
                print("Stopping after having reached %r nonlinear iterations." % self.number_of_iterations)
                self.reached_max_it = True
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

    def __init__(self, macro_grid, element_orders, variable_names=['f']):
        self.macro_nodes = macro_grid
        self.element_orders = numpy.atleast_1d(element_orders)
        self.number_of_elements = len(self.element_orders)
        self.list_of_variables = variable_names
        self.number_of_variables = len(self.list_of_variables)

        self.setGM1d()

        self.quadrature_weights = []
        self.quadrature_points = []
        self.Jx = []
        self.Dx = []
        self.x = numpy.zeros(self.dof1v)
        self.long_quadrature_weights = []
        self.pos = []
        for el_ in range(self.number_of_elements):
            lower_element_boundary = self.macro_nodes[el_]
            upper_element_boundary = self.macro_nodes[el_+1]
            self.Jx.append((upper_element_boundary - lower_element_boundary) / 2.0)
            self.Dx.append(lagrange_derivative_matrix_gll(self.element_orders[el_]+1) / self.Jx[el_])
            qx, qw = gll(self.element_orders[el_]+1, lower_element_boundary, upper_element_boundary)
            self.quadrature_points.append(qx)    # x coordinate (quad points)
            self.quadrature_weights.append(qw)    # quadrature weights
            long_qw = numpy.tile(qw, self.number_of_variables)
            self.long_quadrature_weights.append(long_qw)
            self.pos.append({})
            for var_ in range(self.number_of_variables):
                first_node_in_element = var_ * len(self.GM1v[el_])
                last_node_in_element = first_node_in_element + len(self.GM1v[el_])
                self.pos[el_][self.list_of_variables[var_]] = numpy.arange(first_node_in_element, last_node_in_element)
            self.x[self.GM1v[el_]] = qx

        self.x = numpy.tile(self.x, self.number_of_variables)

    def plot_mesh(self):
        # Plot nodes and line
        xMicro = self.x
        xMacro = self.macro_nodes
        matplotlib.pyplot.plot((xMacro[0], xMacro[-1]), (0, 0), 'r--', linewidth=2.0)    # Lines
        matplotlib.pyplot.plot(xMicro, xMicro*0, 'ro')    # Nodes (micro)
        matplotlib.pyplot.plot(xMacro, xMacro*0, 'bs', markersize=10)    # Nodes (macro)

        # Plot node and element numbers
        for node_ in range(self.dof1v):
            matplotlib.pyplot.text(self.x[node_], -0.1, str(node_), fontsize=10, color='red')
        for border_ in range(len(xMacro)-1):
            element_center = ( xMacro[border_] + xMacro[border_+1] ) / 2.0
            matplotlib.pyplot.text(element_center, +0.1, str(border_), fontsize=15, color='blue')

        # Write annotations
        first_element_center = ( xMacro[0] + xMacro[1] ) / 2.0
        matplotlib.pyplot.annotate('element numbers', xy=(first_element_center, 0.17), xytext=(first_element_center, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
        matplotlib.pyplot.annotate('node number', xy=(xMicro[1], -0.12), xytext=(xMicro[1], -0.3), arrowprops=dict(facecolor='black', shrink=0.05))
        matplotlib.pyplot.text((xMacro[0]+xMacro[-1])/4.0, -0.9, 'Degrees of freedom per variable: %d\nTotal degrees of freedom: %d\nNumber of elements: %d\nVariables: %d %s' %(self.dof1v, self.dofNv, self.number_of_elements, self.number_of_variables, self.list_of_variables))
        matplotlib.pyplot.title('1-D mesh information')
        matplotlib.pyplot.xlabel("Independent variable coordinate")
        matplotlib.pyplot.axis([xMacro[0], xMacro[-1], -1, 1])
        matplotlib.pyplot.show()

    def setGM1d(self):
        self.dof1v = numpy.sum(self.element_orders)+1
        self.dofNv = self.dof1v * self.number_of_variables

        nodeCnt1v_ = 0
        self.GM = []
        self.GM1v = []
        for el_ in range(self.number_of_elements):
            elementSize = self.number_of_variables * (self.element_orders[el_]+1)
            self.GM.append(numpy.zeros(elementSize, dtype=numpy.int))
            for var_ in range(self.number_of_variables):
                firstVarPos = var_ * (self.element_orders[el_]+1)
                lastVarPos = firstVarPos + self.element_orders[el_]
                firstVarNumber = var_ * self.dof1v + nodeCnt1v_
                lastVarNumber = firstVarNumber + self.element_orders[el_]
                self.GM[el_][firstVarPos : lastVarPos + 1] = numpy.arange(firstVarNumber, lastVarNumber + 1)
            nodeCnt1v_ += self.element_orders[el_]
            self.GM1v.append( self.GM[el_][0 : self.element_orders[el_]+1] )


