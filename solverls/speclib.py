import math    # for math.sqrt() in conjGrad()

import numpy


# ********************************************************** #
# ********************** LIBRARY CODE ********************** #
# ********************************************************** #

def conj_grad_elem(k_elem, Ge, gm, dof, x=None, TOL=1.0e-12):
    """
    Attempts to solve the system of linear equations A*x=b for x. The dof-by-dof coefficient matrix A must be symmetric and positive definite (SPD), and should also be large and sparse. 
    The matrix A is only filled across squared blocks on its main diagonal, as results from a finite-element problem. The list Ke contains the non-empty blocks on matrix A, whose lengths need to match the lengths of the vectors in Ge. The list in gm is the gathering matrix specifying the position of Ke blocks in the matrix A and Ge blocks in vector b.
                   
    Syntax: x = conjGradElem(Ke, Ge, gm, dof, x=None, TOL=1.0e-9)
    <list>    Ke = list of dof-by-dof <numpy.ndarray> matrices (must be SPD)
    <list>    Ge = a list of dof-length <numpy.array> vectors
    <list>    gm = a list of the nodes belonging to each element
    <int>    dof = degrees of freedom of the full system of equations
    <numpy.array>    x = initial iteration value for solution (default is zeros)
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """

    if x is None:
        x = numpy.zeros(dof)

    number_of_elements = len(gm)

    # r = loc2gbl(Ge, gm, dof) - loc2gblMatrixVector(Ke, gbl2loc(x, gm), gm, dof)
    r = numpy.zeros(dof)
    for el_ in range(number_of_elements):
        mat_vec_local_product = k_elem[el_].dot(x[gm[el_]])
        local_index = numpy.arange(len(gm[el_]))
        r[gm[el_]] += Ge[el_][local_index] - mat_vec_local_product[local_index]

    s = r.copy()
    for i in range(dof):

        # u = loc2gblMatrixVector(Ke, gbl2loc(s, gm), gm, dof)
        u = numpy.zeros(dof)
        for el_ in range(number_of_elements):
            r_elem = k_elem[el_].dot(s[gm[el_]])
            local_index = numpy.arange(len(gm[el_]))
            u[gm[el_]] += r_elem[local_index]

        alpha = s.dot(r) / s.dot(u)
        x = x + alpha * s

        # r = loc2gbl(Ge, gm, dof) - loc2gblMatrixVector(Ke, gbl2loc(x, gm), gm, dof)
        r = numpy.zeros(dof)
        for el_ in range(number_of_elements):
            mat_vec_local_product = k_elem[el_].dot(x[gm[el_]])
            local_index = numpy.arange(len(gm[el_]))
            r[gm[el_]] += Ge[el_][local_index] - mat_vec_local_product[local_index]

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
        w[i] = 2.0 / ((np-1) * np * L ** 2.0)

    return p, w


def gll(number_of_points, x_min, x_max):
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

    quadrature_points, quadrature_weights = gauss_lobatto_legendre(number_of_points)
    delta = x_max - x_min

    # mapping from (-1,1) -> (x_min, x_max)
    for i in range(number_of_points):
        quadrature_points[i] = delta / 2.0 * (quadrature_points[i]+1.0) + x_min
        quadrature_weights[i] *= delta / 2.0

    return quadrature_points, quadrature_weights


def lagrange_derivative_matrix_gll(np):
    """
    Returns a matrix containing the values of the derivatives of the Lagrange polynomials l'_j evaluated at the GLL quadrature points x_i of order np-1, where [-1 <= x_i <= 1]. The obtained matrix (numpy.ndarray) is defined as: D_{ij} = l'_j(x_i) for i,j = 0:np-1
                   
    Syntax: D = lagrange_derivative_matrix_gll(np, x)
    <int>    np = number of points
    
    Author    : Alfredo R. Carella <alfredocarella@gmail.com>
    """

    gll_derivative_matrix = numpy.zeros((np, np))
    points, weights = gauss_lobatto_legendre(np)

    for i in range(np):
        for j in range(np):

            if i == j:
                pass    # D[i,j]=0 for the main diagonal
            else:
                # Eq. 4.34 in DeMaerschalck2003
                gll_derivative_matrix[i, j] = legendre_polynomial(np-1, points[i]) / (legendre_polynomial(np-1, points[j]) * (points[i] - points[j]))

    gll_derivative_matrix[0, 0] = -np * (np-1) / 4.0
    gll_derivative_matrix[np-1, np-1] = np * (np-1) / 4.0

    return gll_derivative_matrix


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
        # Recurrence 4.5.10 in 'Press1993.pdf'
        for i in range(1,n):
            lagrange_polynomial[i+1] = (2.0*i+1)/(i+1.0)*x*lagrange_polynomial[i] - i/(i+1.0)*lagrange_polynomial[i-1]
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

    lagrange_polynomial = numpy.zeros((n+1, len(numpy.atleast_1d(x))))
    lagrange_polynomial[0, :] = 1.0
    lagrange_polynomial[1, :] = x

    # Recurrence 4.5.10 in 'Press1993.pdf'
    if n > 1:
        for i in range(1, n):
            lagrange_polynomial[i+1, :] = ((2.0 * i+1.0) / (i+1.0) * x) * lagrange_polynomial[i, :] - i / (i+1.0)*lagrange_polynomial[i-1, :]
    else:
        pass

    return lagrange_polynomial[n, :]


