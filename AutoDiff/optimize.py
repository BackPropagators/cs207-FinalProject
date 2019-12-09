from AutoDiff.ForwardAD import Var, MultiFunc
import numpy as np
import types

def optimize(func, initial_guess, tolerance = 10e-6, solver = 'BFGS', max_iter = 100,
             gd_lr = 0.1):
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
        solver : string specifying teh solver to be used. Options are gradient descent 'GD'
                 and the Broyden Fletcher Goldfarb Shanno method 'BFGS'
                 by default equal to 'BFGS'
        max_iter : integer corresponding to the maximum number of iterations  for the algorithm
                   by default equal to 100
        gd_lr : floating point number, learning rate as used for gradient descent
                    by default equal to 0.1
                    Note that this parameter will only be used when solver is specified to 'GD'


        SOLVER SPECIFICS
        =======

        1. Gradient Descent

            - start with initial guess x_0
            - while convergence criterium not met for x_k:
                compute gradient
                update x_(k+1) = x_k - learning_rate*gradient
                x_k = x_(k+1)
            - return result

        2. BFGS

            - start with initial guess x_0 and B_0 = indentity matrix
            - while convergence criterium not met for x_k:
                compute gradient
                solve linear system B_k*s_k = - gradient for step s_k
                compute x_(k+1) = x_k + s_k
                update x_k = x_(k+1)

                compute gradient at x_k
                y_k = new_gradient - old_gradient
                compute Delta_B = (y_k*y_k.T)/(y_k.T*s_k) - (B_k*s_k*s_k.T*B)/(s_k.T*B_k*s_k)
                compute B_(k+1) = B_k + Delta_B
                update B_k = B_(k+1)

            - return result

            Note that B_k tends to approach the Hessian for multiple iterations, so approaches the Newton's method
            Better performance than gradient descent, but worse than Newton's method is expected

        RETURNS
        =======
        opt_val : list of optimal values for the input of func, minimizing func
        minimum : function func evaluated at the optimal values in opt_val
        tol : the tolerance achieved

        EXAMPLES
        =======

        ValueError is raised when solver is not 'GD' or 'BFGS'
        """

        if not isinstance(func, types.FunctionType):
            raise TypeError('Func must be a callable function.')

        # create the var objects needed
        var_list = []
        for val in initial_guess:
            var_list.append(Var(val))

        if solver == 'GD':
            n_iter = 0

            # get current evaluation of f at initial guess
            if len(var_list) == 1:
                current = func(var_list[0])
            else:
                current = func(var_list)

            # get gradient
            current_der = current.get_der(var_list)
            current_val = current.get_value()
            norm_der = np.linalg.norm(current_der)

            while n_iter <= max_iter and norm_der >= tolerance:
                if len(var_list) == 1:
                    current = func(var_list[0])
                else:
                    current = func(var_list)
                current_der = current.get_der(var_list)
                current_val = current.get_value()

                for i, var in enumerate(var_list):
                    var._set_value((var - gd_lr*current_der[i]).get_value())

                norm_der = np.linalg.norm(current_der)

                n_iter += 1

            opt_val = []
            for var in var_list:
                opt_val.append(var.get_value())

            return opt_val, current_val, n_iter, norm_der

        elif solver == 'BFGS':
            n_iter = 0

            # get current evaluation of f at initial guess
            if len(var_list) == 1:
                current = func(var_list[0])
            else:
                current = func(var_list)

            # get gradient
            current_der = current.get_der(var_list)
            current_val = current.get_value()
            norm_der = np.linalg.norm(current_der)

            # initialize B
            B = np.identity(len(var_list))

            while n_iter <= max_iter and norm_der >= tolerance:
                if len(var_list) == 1:
                    current = func(var_list[0])
                else:
                    current = func(var_list)
                current_der = current.get_der(var_list)
                current_val = current.get_value()

                B_inv = np.linalg.inv(B)
                step = np.dot(B_inv, -np.array(current_der)).reshape(len(var_list),1)

                for i, var in enumerate(var_list):
                    var._set_value((var + step[i][0]).get_value())

                # compute new B
                if len(var_list) == 1:
                    new = func(var_list[0])
                else:
                    new = func(var_list)
                new_der = new.get_der(var_list)
                y_k = np.array(new_der).reshape(len(var_list),1) - np.array(current_der).reshape(len(var_list),1)
                delta_B_1 = (y_k*y_k.T)/np.dot(y_k.T,step)[0]
                delta_B_2 = np.dot(np.dot(B,step),np.dot(step.T,B))/np.dot(step.T, np.dot(B,step))[0][0]
                B = B + delta_B_1 - delta_B_2

                norm_der = np.linalg.norm(step)

                n_iter += 1

            opt_val = []
            for var in var_list:
                opt_val.append(var.get_value())

            return opt_val, current_val, n_iter, norm_der


        else:
            raise ValueError('No other solvers are built-in. Please choose GD or the default BFGS')


x = Var(1)
y = Var(1)
z = Var(1)
var_list = [x,y,z]

def f(vars):
    x,y = vars
    return (x-1)**2 - Var.sin(y)**4

print(optimize(f, [2,1], solver = 'BFGS', tolerance=10e-8, max_iter=200, gd_lr=0.2))
