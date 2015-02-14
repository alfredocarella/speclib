import math
import numpy

__author__ = 'Alfredo Carella'

# ********************************************************** #
# ********************** LIBRARY CODE ********************** #
# ********************************************************** #
#TODO: Remove 'dof' parameter from conj_grad_elem(...), it can be calculated from 'gm'


def conj_grad_elem(k_elem, g_elem, gm, dof, x=None, tol=1.0e-12):
    """
    Attempts to solve the linear system Ax=b. The SPD matrix A should be sparse.
    Matrices resulting from a finite-element problem are only filled across squared blocks on its main diagonal.
    The list Ke contains the non-empty blocks on matrix A, whose lengths need to match the lengths of the vectors in
    Ge. The list in gm is the gathering matrix specifying the position of Ke blocks in the matrix A and Ge blocks in
    vector b.
                   
    Syntax: x = conjGradElem(Ke, Ge, gm, dof, x=None, TOL=1.0e-9)
    Ke = list of dof-by-dof <numpy.ndarray> matrices (must be SPD)
    Ge = a list of dof-length <numpy.array> vectors
    gm = a list of the nodes belonging to each element
    dof = degrees of freedom of the full system of equations
    <numpy.array>    x = initial iteration value for solution (default is zeros)
    """

    if x is None:
        x = numpy.zeros(dof)

    number_of_elements = len(gm)

    # r = loc2gbl(Ge, gm, dof) - loc2gblMatrixVector(Ke, gbl2loc(x, gm), gm, dof)
    r = numpy.zeros(dof)
    for el_ in range(number_of_elements):
        mat_vec_local_product = k_elem[el_].dot(x[gm[el_]])
        local_index = numpy.arange(len(gm[el_]))
        r[gm[el_]] += g_elem[el_][local_index] - mat_vec_local_product[local_index]

    # s = r.copy()
    s = r
    cg_iteration = 0

    for cg_iteration in range(dof):
        u = numpy.zeros(dof)
        for el_ in range(number_of_elements):
            r_elem = k_elem[el_].dot(s[gm[el_]])
            local_index = numpy.arange(len(gm[el_]))
            u[gm[el_]] += r_elem[local_index]

        alpha = s.dot(r) / s.dot(u)
        x = x + alpha * s

        r = numpy.zeros(dof)
        for el_ in range(number_of_elements):
            mat_vec_local_product = k_elem[el_].dot(x[gm[el_]])
            local_index = numpy.arange(len(gm[el_]))
            r[gm[el_]] += g_elem[el_][local_index] - mat_vec_local_product[local_index]

        if math.sqrt(r.dot(r)) < tol:
            break
        else:
            beta = -r.dot(u) / s.dot(u)
            s = r + beta * s

    return x, cg_iteration


def conj_grad(a, b, x=None, tol=1.0e-12):
    """
    Attempts to solve the linear system Ax=b. The SPD matrix A should be sparse.
                   
    a = n-by-n matrix (must be SPD)
    b = n-length vector
    x = initial iteration value for solution (default is zeros)
    """

    if x is None:
        x = numpy.zeros(len(b))

    n = len(b)
    r = b - a.dot(x)
    s = r.copy()
    cg_iteration = 0
    for cg_iteration in range(n):
        u = a.dot(s)
        alpha = s.dot(r) / s.dot(u)
        x = x + alpha * s
        r = b - a.dot(x)

        if math.sqrt(r.dot(r)) < tol:
            break
        else:
            beta = -r.dot(u) / s.dot(u)
            s = r + beta * s

    return x, cg_iteration


def gauss_legendre(np):
    """
    Returns a tuple of lists with points and weights for the Gauss Legendre quadrature for the interval [-1, 1].
                   
    Syntax: p, w = gauss_legendre(np)
    np = number of points
    p = quadrature points
    w = quadrature weight
    """

    # This part finds the A-matrix
    a = numpy.zeros((np, np))
    a[0, 1] = 1.0

    if np > 2:
        for i in range(1, np-1):
            a[i, i-1] = i / (2.0*i + 1.0)
            a[i, i+1] = (i+1) / (2.0*i + 1.0)
    else:
        pass

    a[np-1, np-2] = (np-1.0)/(2.0*np-1.0)

    # The array of the sorted eigenvalues/zeros
    p = numpy.sort(numpy.linalg.eigvals(a))

    # This loop finds the associated weights
    w = numpy.zeros(np)
    for j in range(0, np):
        w[j] = 2.0/((1 - p[j]**2.0) * (legendre_derivative(np, p[j]))**2.0)
    return p, w


def gll(number_of_points, x_min=-1.0, x_max=1.0, tol=1e-14):
    """
    Returns a tuple of lists with points and weights for the Gauss Lobatto Legendre quadrature for the interval
    [x_min, x_max].

    Syntax: p, w = gll(np, x_min, x_max)
    np = number of points
    x_min = left interval boundary
    x_max = right interval boundary
    p = quadrature points
    w = quadrature weight
    """

    max_order = number_of_points-1
    quad_points = numpy.zeros(number_of_points)
    quad_points[0], quad_points[-1] = -1.0, 1.0

    quad_weights = numpy.zeros(number_of_points)

    if number_of_points > 3:
        # These points are needed as start (seed) values for the Newton iteration
        gl_points, gl_weights = gauss_legendre(max_order)
        start_values = numpy.zeros(number_of_points)
        start_values[1:max_order] = (gl_points[0:max_order-1] + gl_points[1:max_order]) / 2.0

        # This loop executes the Newton-iteration to find the GLL points
        for i in range(1, max_order):
            quad_points[i] = start_values[i]
            p_old = 0.0
            while abs(quad_points[i] - p_old) > tol:
                p_old = quad_points[i]
                l = legendre_polynomial(max_order, p_old)
                dl = legendre_derivative(max_order, p_old)
                quad_points[i] = p_old + ((1.0 - p_old ** 2.0) * dl) / (max_order * number_of_points * l)

    # This loop finds the associated weights
    for i in range(number_of_points):
        l = legendre_polynomial(max_order, quad_points[i])
        quad_weights[i] = 2.0 / (max_order * number_of_points * l ** 2.0)

    # Mapping [-1,1] -> [x_min, x_max]
    if (x_min is not None) and (x_max is not None):
        delta = x_max - x_min
        quad_points = delta / 2.0 * (quad_points + 1.0) + x_min
        quad_weights *= delta / 2.0

    return quad_points, quad_weights


def lagrange_derivative_matrix_gll(np):
    """
    Returns a matrix containing the values of the derivatives of the Lagrange polynomials l'_j evaluated at the GLL
    quadrature points x_i of order np-1, where [-1 <= x_i <= 1]. The obtained matrix (numpy.ndarray) is defined as:
    D_{ij} = l'_j(x_i) for i,j = 0:np-1
                   
    Syntax: D = lagrange_derivative_matrix_gll(np, x)
    np = number of points
    """

    gll_derivative_matrix = numpy.zeros((np, np))
    points, weights = gll(np)

    for i in range(np):
        for j in range(np):

            if i == j:
                pass    # D[i,j]=0 for the main diagonal
            else:
                # Eq. 4.34 in "DeMaerschalck2003"
                gll_derivative_matrix[i, j] = legendre_polynomial(np-1, points[i]) / \
                    (legendre_polynomial(np-1, points[j]) * (points[i] - points[j]))

    gll_derivative_matrix[0, 0] = -np * (np-1) / 4.0
    gll_derivative_matrix[np-1, np-1] = np * (np-1) / 4.0

    return gll_derivative_matrix


def lagrange_interpolating_matrix(x_in, x_out):
    """
    Returns a matrix 'L' that yields 'f(x_out) = L f(x_in)' via Lagrange interpolation.
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
                    interpolating_matrix[k_coord, i_basis] *= (x_out[k_coord] - x_in[j_basis]) / \
                                                              (x_in[i_basis] - x_in[j_basis])

    return interpolating_matrix    # Interpolating matrix


def legendre_derivative(n, x):
    """
    Returns the the value of the derivative of the n_th Legendre polynomial evaluated at the coordinate x. The input
    'x' can be a vector of points.

    n = polynomial order 0,1,...
    x = coordinate -1 =< x =< 1
    """

    lagrange_polynomial = numpy.zeros(n+1)
    lagrange_polynomial[0] = 1.0
    lagrange_polynomial[1] = x

    if x == -1 or x == 1:
        lagrange_derivative = x**(n-1.0) * (1.0/2.0) * n * (n+1.0)
    else:
        # Recurrence 4.5.10 in 'Press1993.pdf'
        for i in range(1, n):
            lagrange_polynomial[i+1] = (2.0*i+1)/(i+1.0)*x*lagrange_polynomial[i] - i/(i+1.0)*lagrange_polynomial[i-1]
        lagrange_derivative = n/(1.0-x**2.0)*lagrange_polynomial[n-1] - n*x/(1.0-x**2.0)*lagrange_polynomial[n]

    return lagrange_derivative


def legendre_polynomial(n, x):
    """
    Returns the value of the n_th Legendre polynomial evaluated at the coordinate 'x'. The input 'x' can be a vector
    of points.

    n = polynomial order 0,1,...
    x = coordinate -1 =< x =< 1
    """

    lagrange_poly = numpy.zeros((n+1, len(numpy.atleast_1d(x))))
    lagrange_poly[0, :] = 1.0
    lagrange_poly[1, :] = x

    # Recurrence 4.5.10 in 'Press1993.pdf'
    if n > 1:
        for i in range(1, n):
            lagrange_poly[i+1, :] = ((2.0*i+1.0) / (i+1.0)*x) * lagrange_poly[i, :] - i / (i+1.0)*lagrange_poly[i-1, :]

    return lagrange_poly[n, :]