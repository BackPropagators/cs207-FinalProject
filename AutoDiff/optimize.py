from ForwardAD import Var, MultiFunc
import numpy as np
import types

def optimize(func, initial_guess, tolerance = 10e-6, solver = 'Newton', max_iter = 100,
             gs_lr = 0.1):
        """
        Returns the values for the elements in var_list that minimize function func.
        The optimization problem is solved with the method solver and accuracy specified by tolerance

        INPUTS
        =======
        func : callable (function)
               - takes in a list of variables
               - returns a single variable
               If its input is a list of Var objects, returns a var object

        initial_guess : list of real numbers, corresponding to the initial guess for optimal values for variables in input of func
        tolerance : floating point number, by default equal to 10e-6
        solver : string specifying teh solver to be used. Options are gradient descent 'GS', Newtont's method 'Newton'
                 and the Broyden Fletcher Goldfarb Shanno method 'BFGS'
                 by default equal to 'Newton'
        max_iter : integer corresponding to the maximum number of iterations  for the algorithm
                   by default equal to 100
        gs_lr : floating point number, learning rate as used for gradient descent
                    by default equal to 0.1
                    Note that this parameter will only be used when solver is specified to 'GS'


        SOLVER SPECIFICS
        =======
        1. Newton

        2. Gradient Descent


        3. BFGS


        RETURNS
        =======
        opt_val : list of optimal values for the input of func, minimizing func
        minimum : function func evaluated at the optimal values in opt_val
        tol : the tolerance achieved

        EXAMPLES
        =======

        ValueError is raised when solver is not 'GS', 'Newton' or 'BFGS'
        """

        if not isinstance(func, types.FunctionType):
            raise TypeError('Func must be a callable function.')

        # create the var objects needed
        var_list = []
        for val in initial_guess:
            var_list.append(Var(val))

        if solver == 'Newton':
            pass

        elif solver == 'GS':
            n_iter = 0

            # get current evaluation of f at initial guess
            current = func(var_list)

            # get gradient
            current_der = current.get_der(var_list)
            current_val = current.get_value()

            norm_der = np.linalg.norm(current_der)

            while n_iter <= max_iter and norm_der >= tolerance:
                current = func(var_list)
                current_der = current.get_der(var_list)
                current_val = current.get_value()

                for i, var in enumerate(var_list):
                    var.set_value((var - gs_lr*current_der[i]).get_value())

                norm_der = np.linalg.norm(current_der)

                n_iter += 1

            opt_val = []
            for var in var_list:
                opt_val.append(var.get_value())

            return opt_val, current_val, n_iter

        elif solver == 'BFGS':
            pass

        else:
            raise ValueError('No other solvers are built-in. Please choose the default Newton, GS or BFGS')

        return

x = Var(1)
y = Var(1)
z = Var(1)
var_list = [x,y,z]

def f(vars):
    x,y = vars
    return (Var.sin(x)-1)**4 + (y-4)**2

print(optimize(f, [0,1], solver = 'GS', max_iter=10000, gs_lr=0.2))
