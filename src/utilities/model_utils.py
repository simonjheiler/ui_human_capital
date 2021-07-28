""" Utilities for numerical modelling.

This modules contains standardized functions for used
throughout the analytical part of the project for
 handling of the numerical routines.

"""
#####################################################
# IMPORTS
#####################################################
import sys

import numpy as np


#####################################################
# PARAMETERS
#####################################################

global optim_options
optim_options = {}

#####################################################
# FUNCTIONS
#####################################################


def optstep(f, x0, f0, g0, d, method=3, MaxIters=100):
    """Solve a one dimensional optimal step length problem

    % Used to compute step lengths in multidimensional optimization
    % USAGE % [s, fx, errcode, iters] = optstep(method, f, x0, f0, g0, d, MaxIters);

    Parameters
    ----------
        f : string
            function name
        x0 : np.float
            starting value
        f0 : np.float
            function value at the starting point
        g0 : np.float
            gradient of f at the starting point
        d :
            the search direction
        method : int
            method to calculate step length; options:
                1 - step length is set to 1
                2 - STEPBHHH
                3 - STEPBT (default)
                4 - STEPGOLD (called if other methods fail)
        MaxIters :
            the maximum  # of itereations before trying something else

    Returns
    -------
        optval : type
            the current value of the option

    Raises
    ------


    Copyright(c) 1997 - 2000, Paul L.Fackler & Mario J.Miranda
    paul_fackler @ ncsu.edu, miranda .4 @ osu.edu

    """

    if method == 1:
        fx = eval(f, x0 + d)
        if fx < f0:
            s = 1
            iters = 1
            errcode = 0
        else:
            (s, fx, iters, errcode) = stepgold(f, x0, f0, g0, d, MaxIters)

    elif method == 2:
        (s, fx, iters, errcode) = stepbhhh(f, x0, f0, g0, d, MaxIters)
        if errcode:
            (s, fx, iters2, errcode) = stepgold(f, x0, f0, g0, d, MaxIters)
            iters = iters + iters2

    elif method == 3:
        (s, fx, iters, errcode) = stepbt(f, x0, f0, g0, d, MaxIters)
        if errcode:
            (s, fx, iters2, errcode) = stepgold(f, x0, f0, g0, d, MaxIters)
            iters = iters + iters2

    elif method == 4:
        (s, fx, iters, errcode) = stepgold(f, x0, f0, g0, d, MaxIters)

    else:
        raise ValueError

    return s, fx, iters, errcode


def stepbhhh(f, x0, f0, g0, d, MaxIters):
    """Compute approximate minimum step length using BHHH algorithm

    Parameters
    ----------
        f : string
            the objective function being minimized
        x0 : float
            the current value of the parameters
        f0 :
            the value of f(x0)
        g0 :
            the gradient vector of f at x0
        d :
            the search direction
        MaxIters : int
            the maximum number of function evaluations allowed

    Returns
    -------
        s: the optimal step in the direction d
        fs: the value of f at x + s * d,
        iter: the number of iterations used

    Raises
    ------
        errcode: equals 1 if maximum iterations are exceeded

    STEPBHHH uses an algorithm based on one discussed in Berndt, et.al., Annals of
    Economic and Social Measurement, 1974, pp. 653 - 665. This procedure specifies a
    cone of convergence in the plane defined by the direction vector, d, and the
    value of the objective % function. The cone is defined by the lines through the
    origin (x, f(x)) with slopes(d'g) * delta and (d'g) * (1-delta).  Delta must lie on
    (0, 0.5). The procedure iterates until a point is found on the objective function
    that lies within the cone. In general, the wider the cone, the faster a "suitable"
    step size will be found. If a trial point lies above the cone the step size will
    be increased and if it lies below the cone the step size is decreased.

    """

    # INITIALIZATIONS
    if len(sys.argv) < 6 or np.isempty(MaxIters):
        MaxIters = 25

    delta = optget("optstep", "bhhhcone", 0.0001)
    dg = -d * g0  # directional derivative
    tol1 = dg * delta
    tol0 = dg * (1 - delta)

    s = 1
    ds = 1

    iteration = None
    temp = None
    fs = None
    errcode = 0

    # first bracket the cone
    for i in range(MaxIters):
        iteration = i
        x = x0 + s * d
        fs = eval(f, x)
        temp = (f0 - fs) / s
        if temp < tol0:
            ds = 2 * ds
            s = s + ds
        else:
            break

    if tol0 <= temp <= tol1:
        return s, fs, iteration, errcode

    ds = ds / 2
    s = s - ds

    # then use bisection to get inside it
    for _ in range(MaxIters):
        iteration += 1
        ds = ds / 2
        x = x0 + s * d
        fs = eval(f, x)
        temp = (f0 - fs) / s
        if temp > tol1:
            s = s - ds
        elif temp < tol0:
            s = s + ds
        else:
            return s, fs, iteration, errcode

    errcode = 1

    return s, fs, iteration, errcode


def stepbt(f, x0, f0, g0, d, MaxIters):
    """Compute approximate minimum step length

    Parameters
    ----------
        f : string
            the objective function being minimized
        x0 :
            the current value of the parameters
        f0 :
            the value of f(x); this is passed as argument to save one function
            evaluation
        g0 : np.array()
            the gradient vector of f at x0
        d : int
            the search direction
        MaxIters : int
            the maximum number of "backsteps" of the step length

    Returns
    -------
        s : np.float
            the optimal step in the direction d
        fs : np.float
            the value of f at x + s * d
        iter : int
            the number of iterations used

    Raises
    ------
        errcode : int
            equals 1 if STEPBT fails to find a suitable step length 2 if cubic
            approximation finds negative root

    STEPBT uses a backtracking method similar to Algorithm 6.3 .5 in Dennis and
    Schnabel, Numerical Methods for Unconstrained Optimization % and Nonlinear
    Equations or LNSRCH in sec 9.7 of Press, et al., Numerical Recipes.The
    algorithm approximates the function with a cubic using the function value
    and derivative at the initial point % and two additional points. It determines
    the minimum of the approximation. If this is acceptable it returns, otherwise
    it uses the current and precious point to form a new approximation. The
    convergence criteria is similar to that discussed in Berndt, et.al.,
    Annals of Economic and Social Measurement, 1974, pp. 653 - 665 (see description
    of BHHHSTEP). The change in the step size is also limited to ensure that
    lb * s(k) <= s(k + 1) <= ub * s(k)(defaults: lb = 0.1, ub = 0.5).

    """

    # initialisations
    delta = 1e-4  # Defines cone of convergence; must be on(0, 1 / 2)
    ub = 0.5  # Upper bound on acceptable reduction in s
    lb = 0.1  # Lower bound on acceptable reduction in s

    dg = -d * g0  # directional derivative

    tol1 = delta * dg
    tol0 = (1 - delta) * dg

    iteration = None
    errcode = 0

    # full step
    s = 1
    fs = eval(f, x0 + d)
    if -fs + f0 <= tol1:
        iteration = 1
        return s, fs, iteration, errcode

    # quadratic approximation
    s2 = s
    fs2 = fs
    s = -0.5 * dg / (-fs + f0 - dg)
    s = max(s, lb)
    fs = eval(f, x0 + s * d)
    temp = (-fs + f0) / s
    if tol0 <= temp & temp <= tol1:
        iteration = 2
        return [s, fs, iter, errcode]

    # cubic approximation
    for i in range(MaxIters):
        iteration += i
        temp = (s - s2) * np.array((s * s, s2 * s2))
        temp = np.array(((-fs + f0 - dg * s), (-fs2 + f0 - dg * s2))) / temp
        a = temp[1] - temp[2]
        b = s * temp(2) - s2 * temp(1)
        s2 = s
        fs2 = fs
        if a == 0:  # quadratic fits exactly
            s = -0.5 * dg / b
        else:
            disc = b * b - 3 * a * dg
            if disc < 0:
                errcode = 2
                return [s, fs, iter, errcode]

    # complex root
    s = (np.sqrt(disc) - b) / (3 * a)

    s = max(min(s, ub * s2), lb * s2)  # ensures acceptable step size
    fs = eval(f, x0 + s * d)
    temp = (-fs + f0) / s
    if tol0 <= temp & temp <= tol1:
        return [s, fs, iter, errcode]

    errcode = 1

    return [s, fs, iter, errcode]


def stepgold(f, x0, f0, g0, d, MaxIters):
    """Compute approximate minimum step length using golden search
    algorithm

    Parameters
    ----------
         f : string
            the objective function being minimized
        x0 :
            the current value of the parameters
        f0 :
            the value of f(x0)
        g0 :
            the gradient vector of f at x0 (note: not used)
        d :
            the search direction
        MaxIters :
            the maximum number of function evaluations allowed

    Returns
    -------
        s :
            the optimal step in the direction d
        fs :
            the value of f at x + s * d
        iter :
            the number of iterations used

    Raises
    ------
        errcode :
            equals 1 if maximum iterations are exceeded

    STEPGOLD uses step doubling to find an initial bracket and then uses the
    golden search method to find a minimum value within the bracket.
    Iterations cease if the bracket is less than TOL or a maximum number of
    iterations is reached.

    """

    alpha1 = (3 - np.sqrt(5)) / 2
    alpha2 = (np.sqrt(5) - 1) / 2
    tol = 1e-4  # tolerance used for Golden search algorithm
    tol = tol * (alpha1 * alpha2)  # the bracket will be len / (alpha1 * alpha2)
    s = 1
    errcode = 1  # 1 if the search is unsuccessful; otherwise 0

    iteration = 0
    s0 = 0

    # Find a bracketing interval
    fs = eval(f, x0 + d)
    if f0 >= fs:
        len = alpha1
    else:
        for _ in range(MaxIters):
            iteration += 1
            s = 2 * s
            fl = fs
            fs = eval(f, x0 + s * d)
            if fs <= fl:
                len = alpha1 * (s - s0)
                break
            else:
                f0 = fl
                s0 = s / 2

    if iteration >= MaxIters:
        s = s / 2
        fs = fl
        return s, fs, iteration, errcode

    xl = x0 + (s0 + len) * d
    xs = x0 + (s - len) * d

    s = s - len
    len = len * alpha2  # len now measures relative distance between xl and xs

    fs = eval(f, xs)
    fl = eval(f, xl)

    # Golden search to find minimum
    while iteration < MaxIters:
        iteration += 1
        if fs < fl:
            s = s - len
            len = len * alpha2
            xs = xl
            xl = xl - len * d
            fs = fl
            fl = eval(f, xl)
        else:
            len = len * alpha2
            s = s + len
            xl = xs
            xs = xs + len * d
            fl = fs
            fs = eval(f, xs)
            if len < tol:
                errcode = 0
                break

    if fl > fs:
        fs = fl
        s = s - len

    return s, fs, iter, errcode


def optget(funcname, optname, *args):
    """Get previously set function default values

    Parameters
    ----------
        funcname : string
            name of function
        optname : string
            name of option
        *args : float
            option value

    Returns
    -------
        optval : float
            the current value of the option

    If there does not yet exist a dictionary entry for
    options[funcname][optname], it will be set to the first
    argument after 'optname'. If the dictionary entry has
    already been set, all arguments after 'optname' have no effect
    and the existing value will be returned. Use 'optset' to
    change a previously set field.

    Copyright (c) 1997-2000, Paul L. Fackler & Mario J. Miranda
    paul_fackler@ncsu.edu, miranda.4@osu.edu

    """

    funcname = np.char.lower(funcname)
    optname = np.char.lower(optname)

    try:
        optvalue = optim_options[funcname][optname]
    except KeyError:
        optvalue = args[0]
        optim_options[funcname][optname] = optvalue

    return optvalue


def optset(funcname, optname, optvalue):
    """Set function options

    Parameters
    ----------
        funcname : string
            name of function
        optname : string
            name of option
        optval : float
            option value

    Returns
    -------
        None

    If optname = 'defaults' the current setting of the options will be cleared.
    The next time the function is called, the default options will be restored.

    Copyright (c) 1997-2002, Paul L. Fackler & Mario J. Miranda
    paul_fackler@ncsu.edu, miranda.4@osu.edu

    """
    optvar = np.char.lower(funcname) + "_options"  # name of option variable
    optname = np.char.lower(optname)  # name of option field

    if optname == "defaults":
        optim_options[optvar] = None  # clears option value
    else:
        optim_options[optvar] = optvalue  # set specified field
