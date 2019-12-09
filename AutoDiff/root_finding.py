from ForwardAD import Var, MultiFunc
import numpy as np
import math

def Newton(func, guess, tol = 10**(-8), max_iter = 2000):
    '''
        Returns the values of the roots of function func
        The optimization problem is solved with the method solver and accuracy specified by tolerance

        INPUTS
        =======
        func : callable (function)
               - takes in a list of variables
               - returns a single variable

        guess : list of real numbers, corresponding to the initial guess for roots of the function func
        tol : the tolerance threshold of errors, floating point number, by default equal to 10e-8
        max_iter : integer corresponding to the maximum number of iterations  for the algorithm
                   by default equal to 2000

        RETURNS
        =======
        xnext: the root of func when the tolerance tol is reached, 
        or the maximum iteration number, max_iter, is reached.

        EXAMPLES
        =======
        # >>> f = lambda x: x**2-1
        # >>> guess = 2
        # >>> root = Newton(f, guess)
          1.0
    '''
    xcur = Var(guess)
    fxcur = func(xcur)
    xnext = xcur - Var(fxcur.get_value()/fxcur.get_der()[0])
    err = abs(xnext.get_value()-xcur.get_value())
    i = 0
    while err > tol:
        xcur = xnext
        fxcur = func(xcur)
        xnext = xcur - Var(fxcur.get_value()/fxcur.get_der()[0])
        err = abs(xnext.get_value()-xcur.get_value())
        i += 1
        if i > max_iter:
            print('The execution goes beyond the maximum number of iterations, \
                  so it is stopped and the most current result is printed')
            break
    return xnext.get_value()


    