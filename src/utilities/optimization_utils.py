"""
Utilities for optimization routines.
"""
#####################################################
# IMPORTS
#####################################################
import warnings

import numpy as np


#####################################################
# PARAMETERS
#####################################################


#####################################################
# FUNCTIONS
#####################################################


def get_step_size(f, x0, f0, g0, d, controls, *args):
    """
    Solve a one dimensional optimal step length problem.

    :parameter
    f : functional
        Objective function of maximization problem.
    x0 : array
        Starting value for step.
    f0 : float
        Value of objective function evaluated at starting value.
    g0 : array
        Gradient of the objective function at starting value.
    d : array
        Search direction vector.
    controls : dict
        Dictionary of function controls (details see below)
    args : tuple
        Additional arguments for objective function.

    :returns:
    s : float
        Optimal step length in direction d
    fx : float
        Value of objective function after optimal step (i.e. at x0 + s * d)
    iteration : int
        Number of iterations conducted in step size calculation
    errcode : bool
        Error flag: TRUE if step method returns error flag
    instr_eq : float
        Equilibrium instrument rate after optimal step (i.e. at x0 + s * d)

    Wrapper function to compute step lengths in multidimensional optimization
    problems.

    Controls (relevant for this routine):
        method : string
            Method to calculate optimal step length. Available options
            - "full" : step length is set to 1
            - "bhhh" : BHHH STEP (currently not implemented)
                    # todo: adjust after implementation
            - "bt" : BT STEP (default)
            - "gold" : GOLD STEP
        n_iterations_step_max : int
            Maximum number of iterations for step length calculation

    Modified from the corresponding file by Paul L. Fackler & Mario J. Miranda
    Copyright(c) 1997 - 2000, Paul L. Fackler & Mario J. Miranda
    paul_fackler@ncsu.edu, miranda.4@osu.edu

    """
    method = controls["step_method"]

    if method == "full":
        fx, instr_eq = f(x0 + d, controls, *args)
        if fx < f0:
            s = 1
            iteration = 1
            errcode = False
            print("full step")
            return s, fx, instr_eq, iteration, errcode
        else:
            s, fx, instr_eq, iteration, errcode = step_size_gold(
                f, x0, f0, g0, d, controls, *args
            )
            print("gold step")
            return s, fx, instr_eq, iteration, errcode

    elif method == "bhhh":
        s, fx, instr_eq, iteration, errcode = step_size_bhhh(
            f, x0, f0, g0, d, controls, *args
        )
        if not errcode:
            print("bhhh step")
            return s, fx, instr_eq, iteration, errcode
        else:
            s, fx, instr_eq, iterations_2, errcode = step_size_gold(
                f, x0, f0, g0, d, controls, *args
            )
            iteration = iteration + iterations_2
            print("gold step")
            return s, fx, instr_eq, iteration, errcode

    elif method == "bt":
        s, fx, instr_eq, iteration, errcode = step_size_bt(
            f, x0, f0, g0, d, controls, *args
        )
        if not errcode:
            print("BT step")
            return s, fx, instr_eq, iteration, errcode
        else:
            s, fx, instr_eq, iterations_2, errcode = step_size_gold(
                f, x0, f0, g0, d, controls, *args
            )
            iteration = iteration + iterations_2
            print("gold step")
            return s, fx, instr_eq, iteration, errcode

    elif method == "gold":
        s, fx, instr_eq, iteration, errcode = step_size_gold(
            f, x0, f0, g0, d, controls, *args
        )
        print("gold step")
        return s, fx, instr_eq, iteration, errcode

    else:
        raise ValueError(
            "Step method unknown; please select one of " "['full', 'bt', 'gold']"
        )


def step_size_bt(f, x0, f0, g0, d, controls, *args):
    """
    Compute approximate minimum step length

    :parameter
    f : functional
        Objective function of the maximization problem
    x0 : array
        Starting point for the current step
    f0 : float
        Value of the objective function at the starting point
    g0 : array
        Gradient vector of the objective function at the starting
        point
    d : array
        Search direction vector
    controls : dict
        Dictionary of function controls (details see below)
    args : tuple
        Additional arguments passed to objective function

    :returns
    s : float
        Optimal step size in direction d
    fs : float
        Value of the objective function after optimal step (i.e. at x0 + s * d)
    iterations : int
        Number of iterations conducted to find optimal step size
    errcode: bool
        Error flag: TRUE if
            - function fails to find a suitable step length, or
            - cubic approximation finds a negative root
    instr_eq : float
        Equilibrium instrument rate after optimal step (i.e. at x0 + s * d)

    *step_size_bt* uses a backtracking method similar to Algorithm 6.3 .5 in
    Dennis and Schnabel, Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations or *LNSRCH* in sec 9.7 of Press, et al.,
    Numerical Recipes. The algorithm approximates the function with a cubic
    using the function value and derivative at the initial point and two
    additional points.It determines the minimum of the approximation. If this
    is acceptable it returns, otherwise it uses the current and previous point
    to form a new approximation. The convergence criteria is similar to that
    discussed in Berndt, et.al., Annals of Economic and Social Measurement,
    1974, pp. 653 - 665 (see description of *step_size_bhhh*).The change in the step
    size is also limited to ensure that

            lb * s(k) <= s(k + 1) <= ub * s(k)

    (defaults: lb = 0.1, ub = 0.5).

    Controls (relevant for this routine):
        n_iterations_step_max : int
            Maximum number of iterations for step length calculation

    """

    # Initializations
    n_iterations_step_max = controls["n_iterations_step_max"]
    delta = 1e-4  # Defines cone of convergence; must be on (0, 1 / 2)
    ub = 0.5  # Upper bound on acceptable reduction in s.
    lb = 0.1  # Lower bound on acceptable reduction in s.
    errcode = False
    dg = np.dot(d, g0)  # directional derivative

    slope_bound_lower = delta * dg
    slope_bound_upper = (1 - delta) * dg
    iteration = 0

    # (I) FULL STEP IN THE DIRECTION OF D
    step = 1
    fs, instr_eq = f(x0 + step * d, controls, *args)
    iteration += 1

    # check if value of objective after step is in cone of convergence
    slope = (fs - f0) / step
    # if slope_bound_lower <= slope <= slope_bound_upper:
    #     return step, fs, instr_eq, iteration, errcode
    if slope_bound_lower <= slope:
        return step, fs, instr_eq, iteration, errcode

    # (II) QUADRATIC APPROXIMATION OF OBJECTIVE FUNCTION
    # AND OPTIMAL STEP IN THE SEARCH DIRECTION

    # use f(x), f'(x) and f(x + s * d) to approximate f
    # with 2nd degree polynomial:
    # f(x + s * d) = h(s) = a * s^2 + b * s + c
    # <=> h is maximal at s = - b / (2a)

    c = f0
    b = dg
    a = (fs - step * b - c) / step ** 2

    # store initial step and value after initial step
    # for use in cubic approximation
    step_2 = step
    fs_2 = fs

    step_1 = -b / (2 * a)
    step_1 = max(step_1, lb)  # ensure lower bound on step length
    fs_1, instr_eq = f(x0 + step_1 * d, controls, *args)
    iteration += 1

    # check if value of objective after step is in cone of convergence
    slope = (fs_1 - f0) / step_1
    if slope_bound_lower <= slope <= slope_bound_upper:
        return step_1, fs_1, instr_eq, iteration, errcode

    # (III) CUBIC APPROXIMATION OF OBJECTIVE FUNCTION
    # AND OPTIMAL STEP IN THE SEARCH DIRECTION

    # (i) use f(x), f'(x), f(x + s1 * d), and f(x + s2 * d) to
    # approximate with 3rd degree polynomial:
    # f(x + s * d) = h(s) = a * s^3 + b * s^2 + c * s + d
    # (ii) check if value of objective function is within cone of
    # convergence after optimal candidate step; if not, update
    # s_2 to s_1 and compute new candidate step

    while iteration < n_iterations_step_max:
        d = f0
        c = dg
        b = step_1 * (fs_2 - c * step_2 - d) / (
            step_2 ** 2 * (step_1 - step_2)
        ) - step_2 * (fs_1 - c * step_1 - d) / (step_1 ** 2 * (step_1 - step_2))
        a = (fs_1 - c * step_1 - d) / (step_1 ** 2 * (step_1 - step_2)) - (
            fs_2 - c * step_2 - d
        ) / (step_2 ** 2 * (step_1 - step_2))

        # store current step and value after current step
        # for use in cubic approximation
        step_2 = step_1
        fs_2 = fs_1

        if a == 0:  # quadratic fits exactly
            step_1 = -c / (2 * b)
        else:
            # optimal step is given by root of first derivative
            # at s = (-b + sqrt( b^2 - 3 * a * c)) / (3 * a)
            # check for complex root in solution of polynomial
            if (b ** 2 - 3 * a * c) < 0:
                errcode = 2
                return step_1, fs_1, instr_eq, iteration, errcode
            else:
                step_1 = (-b + np.sqrt(b ** 2 - 3 * a * c)) / (3 * a)

        # ensure acceptable step size
        step_1 = max(min(step_1, ub * step_2), lb * step_2)
        fs_1, instr_eq = f(x0 + step_1 * d, controls, *args)
        iteration += 1

        # check if value of objective after step is in cone of convergence
        slope = (fs_1 - f0) / step_1
        if slope_bound_lower <= slope <= slope_bound_upper:
            return step_1, fs_1, instr_eq, iteration, errcode

    if iteration == n_iterations_step_max:
        warnings.warn("maximum number of step size iterations " "in BT search reached")
        errcode = True

    return step_1, fs_1, instr_eq, iteration, errcode


def step_size_bhhh(f, x0, f0, g0, d, controls, *args):
    """
    Compute an approximate minimum step length

    :parameter
    f : functional
        Objective function of the maximization problem
    x0 : array
        Starting point for the current step
    f0 : float
        Value of the objective function at the starting point
    g0 : array
        Gradient vector of the objective function at the starting
        point (note: not used)
    d : array
        Search direction vector
    controls : dict
        Dictionary of function controls (details see below)
    args : tuple
        Additional arguments passed to objective function

    :returns
    s : float
        Optimal step size in direction d
    fs : float
        Value of the objective function after optimal step (i.e. at x0 + s * d)
    iterations : int
        Number of iterations conducted to find optimal step size
    errcode: bool
        Error flag: TRUE if maximum iterations are exceeded
    instr_eq : float
        Equilibrium instrument rate after optimal step (i.e. at x0 + s * d)

    *step_size_bhhh* ...  # todo: complete

    Controls (relevant for this routine):
        n_iterations_step_max : int
            Maximum number of iterations for step length calculation

    """

    pass


def step_size_gold(f, x0, f0, g0, d, controls, *args):
    """
    Compute an approximate minimum step length

    :parameter
    f : functional
        Objective function of the maximization problem
    x0 : array
        Starting point for the current step
    f0 : float
        Value of the objective function at the starting point
    g0 : array
        Gradient vector of the objective function at the starting
        point (note: not used)
    d : array
        Search direction vector
    controls : dict
        Dictionary of function controls (details see below)
    args : tuple
        Additional arguments passed to objective function

    :returns
    s : float
        Optimal step size in direction d
    fs : float
        Value of the objective function after optimal step (i.e. at x0 + s * d)
    iterations : int
        Number of iterations conducted to find optimal step size
    errcode: bool
        Error flag: TRUE if maximum iterations are exceeded
    instr_eq : float
        Equilibrium instrument rate after optimal step (i.e. at x0 + s * d)

    *step_size_gold* uses step doubling to find an initial bracket and then uses
    the golden search method to find a maximum value within the bracket.
    Iterations cease if the bracket is less than TOL or a maximum number
    of iterations is reached.

    Controls (relevant for this routine):
        n_iterations_step_max : int
            Maximum number of iterations for step length calculation

    """

    # load controls
    n_iterations_step_max = controls["n_iterations_step_max"]
    tol = 1e-4  # tolerance used for Golden search algorithm

    errcode = True  # TRUE if the search is unsuccessful; otherwise FALSE

    # initiate bracket search
    bound_lower = 0
    bound_upper = 1
    f_lower = f0
    f_upper, instr_eq = f(x0 + d, controls, *args)
    iteration = 1

    if f_lower <= f_upper:  #
        while iteration < n_iterations_step_max:
            # bound_lower = bound_upper
            # f_lower = f_upper

            bound_upper *= 2
            f_upper, instr_eq = f(x0 + bound_upper * d, controls, *args)

            iteration += 1

            if f_lower >= f_upper:
                break

        # stop step size calculation if maximum number of iterations exceeded
        if iteration == n_iterations_step_max:
            warnings.warn("Maximum number of iterations exceeded in bracket search.")
            step_out = bound_upper
            f_out = f_upper
            instr_eq_out = instr_eq
            return step_out, f_out, instr_eq_out, iteration, errcode

    # once initial bracket has been found, start golden section search
    # compute auxiliary variables
    alpha_1 = (3 - np.sqrt(5)) / 2
    alpha_2 = (np.sqrt(5) - 1) / 2
    tol = tol * (alpha_1 * alpha_2)

    # initiate first iteration
    step_1 = bound_lower + alpha_1 * (bound_upper - bound_lower)
    step_2 = bound_lower + alpha_2 * (bound_upper - bound_lower)
    dist = bound_upper - bound_lower

    f_1, instr_eq_1 = f(x0 + step_1 * d, controls, *args)
    f_2, instr_eq_2 = f(x0 + step_2 * d, controls, *args)
    iteration += 2

    # Golden search to find minimum
    while iteration < n_iterations_step_max:

        if f_1 > f_2:
            dist = dist * alpha_2

            bound_lower = bound_lower
            bound_upper = step_2

            step_2 = step_1
            step_1 = bound_lower + alpha_1 * dist

            f_2 = f_1
            f_1, instr_eq_1 = f(x0 + step_1 * d, controls, *args)

            iteration += 1

        elif f_1 < f_2:
            dist = dist * alpha_2

            bound_lower = step_1
            bound_upper = bound_upper

            step_1 = step_2
            step_2 = bound_lower + alpha_2 * dist

            f_1 = f_2
            f_2, instr_eq_2 = f(x0 + step_2 * d, controls, *args)

            iteration += 1

        else:
            warnings.warn("objective function value identical at both interior points")
            break

        if dist < tol:
            print("minimum bracket size reached")
            errcode = False
            break

    if f_1 > f_2:
        step_out = step_1
        f_out = f_1
        instr_eq_out = instr_eq_1
    else:
        step_out = step_2
        f_out = f_2
        instr_eq_out = instr_eq_2

    return step_out, f_out, instr_eq_out, iteration, errcode
