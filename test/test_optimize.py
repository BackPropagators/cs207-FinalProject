from AutoDiff.optimize import optimize
from AutoDiff.ForwardAD import Var, MultiFunc
import numpy as np

def test_BFGS_1():

    def suite_scalar_input():

        # one variables
        for initial_guess in [-1, 0, 1]:
            def f(vars):
                x = vars
                return x ** 2 + 1
            opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess],
                                                              solver = "BFGS", max_iter=10000, gd_lr=0.2)

            assert np.allclose(opt_val, [0])
            assert np.allclose(current_val, [1])

    def suite_multivariate_input():

        # two variables
        for initial_guess_x, initial_guess_y in [[-3., -3], [1., 1.], [3., 3.]]:
            def f(vars):
                x, y = vars
                return x ** 2 + y ** 2 + 2

            opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess_x,
                                                                  initial_guess_y], solver="BFGS",
                                                              max_iter=10000, gd_lr=0.2)
            assert np.allclose(opt_val, [0, 0])
            assert np.allclose(current_val, [2])


        ## three variables
        for initial_guess_x, initial_guess_y, initial_guess_z in [[-3., -3, 2], [1., 1., 2], [3., 3., 2]]:

            def f(vars):
                x, y, z = vars
                return x ** 2 + y ** 2 + z ** 2 + 3

            opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess_x,
                                                                  initial_guess_y,
                                                                  initial_guess_z], solver="BFGS",
                                                              max_iter=10000, gd_lr=0.2)
            assert np.allclose(opt_val, [0, 0, 0])
            assert np.allclose(current_val, [3])


    suite_scalar_input()
    suite_multivariate_input()

def test_BFGS_2():
    # Rosenbrock function: (a-x)^2 + b(y-x^2)^2 with
    # global minimum at (x, y) = (a, a^2) where f(x,y) = 0

        for initial_guess_x, initial_guess_y in [[-3., -3], [1., 1.], [3., 3.]]:

            def f(vars):
                x, y = vars
                return (3 - x) ** 2 + 2 * (y - x ** 2) ** 2

            opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess_x, initial_guess_y],
                                                              solver="BFGS", max_iter=10000, gd_lr=0.2)

            assert np.allclose(opt_val, [3, 9])
            assert np.allclose(current_val, [0])


def test_BFGS_3():
    # Easom function: -cos(x)·cos(y)·exp(-((x-pi)^2+(y-pi)^2))
    # global minimum at (x,y)=(pi,pi) where f(x,y)=-1

    for initial_guess_x, initial_guess_y in [[2.5, 2.5], [3, 3]]:
        def f(vars):
            x, y = vars
            return -Var.cos(x)*Var.cos(y)*Var.exp(-((x - np.pi)**2 + (y - np.pi)**2))

        opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess_x, initial_guess_y],
                                                          solver="BFGS", max_iter=10000, gd_lr=0.2)

        assert np.allclose(opt_val, [np.pi/2, np.pi/2])


def test_BFGS_4():
    # Himmelblau's function: (x^2 + y - 11)^2 + (x + y^ 2 - 7)^2
    # one local maximum at (x,y) = (-0.270845,-0.923039) with f(x,y)=181.617
    # four local minimum at (x,y) = (3,2) or (-2.805118, 3.131312) or
    #                               (-3.779310, -3.283186) or (3.584428, -1.848126) with f(x,y)=0
    for initial_guess_x, initial_guess_y in [[-3, -3], [0, 0], [1, 1], [3, 3], [0, -1]]:
        def f(vars):
            x, y = vars
            return (x ** 2 + y - 11)**2 + (x + y ** 2 - 7) ** 2

        opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess_x, initial_guess_y],
                                                          solver="BFGS", max_iter=10000, gd_lr=0.2)

        assert ((np.allclose(opt_val, [3, 2]) and np.allclose(current_val, [0])) or
                (np.allclose(opt_val, [2.805118, 3.131312]) and np.allclose(current_val, [0])) or
                (np.allclose(opt_val, [-3.779310, -3.283186]) and np.allclose(current_val, [0])) or
                (np.allclose(opt_val, [3.584428, -1.848126]) and np.allclose(current_val, [0])) or
                (np.allclose(opt_val, [-0.270845, -0.923039]) and np.allclose(current_val, [181.617])))


def test_GD_1():

    def suite_scalar_input():

        # one variables
        for initial_guess in [-1, 0, 1]:
            def f(vars):
                x = vars
                return x ** 2 + 1
            opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess],
                                                              solver = "GD", max_iter=10000, gd_lr = 0.2)

            assert np.allclose(opt_val, [0])
            assert np.allclose(current_val, [1])

    def suite_multivariate_input():

        # two variables
        for initial_guess_x, initial_guess_y in [[-3., -3], [1., 1.], [3., 3.]]:
            def f(vars):
                x, y = vars
                return x ** 2 + y ** 2 + 2

            opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess_x,
                                                                  initial_guess_y], solver="GD",
                                                              max_iter=10000, gd_lr=0.2)
            assert np.allclose(opt_val, [0, 0])
            assert np.allclose(current_val, [2])


        ## three variables
        for initial_guess_x, initial_guess_y, initial_guess_z in [[-3., -3, 2], [1., 1., 2], [3., 3., 2]]:

            def f(vars):
                x, y, z = vars
                return x ** 2 + y ** 2 + z ** 2 + 3

            opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess_x,
                                                                  initial_guess_y,
                                                                  initial_guess_z], solver="GD",
                                                              max_iter=10000, gd_lr=0.2)
            assert np.allclose(opt_val, [0, 0, 0])
            assert np.allclose(current_val, [3])


    suite_scalar_input()
    suite_multivariate_input()


def test_GD_2():
    # Rosenbrock function: (a-x)^2 + b(y-x^2)^2 with
    # global minimum at (x, y) = (a, a^2) where f(x,y) = 0

        for initial_guess_x, initial_guess_y in [[-3., -3], [1., 1.], [3., 3.]]:

            def f(vars):
                x, y = vars
                return x ** 2 + (y - x ** 2) ** 2

            opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess_x, initial_guess_y],
                                                              solver="GD", max_iter=10000, gd_lr=0.02)

            assert np.allclose(opt_val, [0, 0])
            assert np.allclose(current_val, [0])

def test_GD_3():
    # Easom function: -cos(x)·cos(y)·exp(-((x-pi)^2+(y-pi)^2))
    # global minimum at (x,y)=(pi,pi) where f(x,y)=-1

    for initial_guess_x, initial_guess_y in [[2.5, 2.5], [3, 3]]:
        def f(vars):
            x, y = vars
            return -Var.cos(x)*Var.cos(y)*Var.exp(-((x - np.pi)**2 + (y - np.pi)**2))

        opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess_x, initial_guess_y],
                                                          solver="GD", max_iter=20000, gd_lr=0.02)

        assert np.allclose(opt_val, [np.pi, np.pi])

def test_GD_4():
    # Himmelblau's function: (x^2 + y - 11)^2 + (x + y^ 2 - 7)^2
    # one local maximum at (x,y) = (-0.270845,-0.923039) with f(x,y)=181.617
    # four local minimum at (x,y) = (3,2) or (-2.805118, 3.131312) or
    #                               (-3.779310, -3.283186) or (3.584428, -1.848126) with f(x,y)=0
    for initial_guess_x, initial_guess_y in [[0, 0], [1, 1], [3, 3]]:
        def f(vars):
            x, y = vars
            return (x ** 2 + y - 11)**2 + (x + y ** 2 - 7) ** 2

        opt_val, current_val, n_iter, norm_der = optimize(f, [initial_guess_x, initial_guess_y],
                                                          solver="GD", max_iter=10000, gd_lr=0.02)

        assert ((np.allclose(opt_val, [3, 2]) and np.allclose(current_val, [0])) or
                (np.allclose(opt_val, [2.805118, 3.131312]) and np.allclose(current_val, [0])) or
                (np.allclose(opt_val, [-3.779310, -3.283186]) and np.allclose(current_val, [0])) or
                (np.allclose(opt_val, [3.584428, -1.848126]) and np.allclose(current_val, [0])) or
                (np.allclose(opt_val, [-0.270845, -0.923039]) and np.allclose(current_val, [181.617])))


