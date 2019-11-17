import AutoDiff.ForwardAd as Var
import numpy as np
import math


def test_scalar_input():

    def suite_negative():
        x1 = Var(np.array([2.0]))
        f1 = -x1
        assert f1.val == [-2.0]
        assert f1.jacobian == [-1.0]

        x2 = Var(np.array([0.0]))
        f2 = -x2
        assert f2.val == [0.0]
        assert f2.jacobian == [-1.0]

        # suite for operator order
        f3 = - x1 / x1
        assert f3.val == [-1.0]
        assert f3.jacobian == [0.0]

    def suite_abs():
        # abs() not differentiable at zero
        with np.testing.assert_raises(ValueError):
            x1 = Var(np.array([0.0]))
            f1 = abs(x1)

        x2 = Var(np.array([5.0]))
        f2 = abs(x2)
        assert f2.val == [5.0]
        assert f2.jacobian == [1.0]

        x3 = Var(np.array([-5.0]))
        f3 = abs(x3)
        assert f3.val == [5.0]
        assert f3.jacobian == [-1.0]

    def suite_constant():
        x = Var(np.array([4.0]), None)
        f = x
        assert f.val == 4.0
        assert f.jacobian == None

    def suite_sin():
        x1 = Var(np.pi)
        f1 = 10e16 * np.sin(x1)
        assert np.round(f1.val, 2) == [12.25]
        assert np.round(f1.jacobian, 2) == [-1.e+17]

        x2 = Var(np.pi * 3 / 2)
        f2 = 10e16 * np.sin(x2)
        assert np.round(f2.val, 2) == [-1.e+17]
        assert np.round(f2.jacobian, 2) == [-18.37]

    def suite_cos():
        x1 = Var(np.pi)
        f1 = 10e16 * np.cos(x1)
        assert np.round(f1.val, 2) == [-1.e+17]
        assert np.round(f1.jacobian, 2) == [-12.25]

        x2 = Var(np.pi * 3 / 2)
        f2 = 10e16 * np.cos(x2)
        assert np.round(f2.val, 2) == [-18.37]
        assert np.round(f2.jacobian, 2) == [1.e+17]

    def suite_tan():
        # tan() not define for multiples of pi/2
        with np.testing.assert_raises(ValueError):
            x0 = Var(np.pi / 2)
            f0 = np.tan(x0)

        x1 = Var(np.pi / 3)
        f1 = np.tan(x1)
        assert np.round(f1.val, 2) == [1.73]
        assert np.round(f1.jacobian, 2) == [4.0]

        x2 = Var(np.pi / 6)
        f2 = np.tan(x2)
        assert np.round(f2.val, 2) == [0.58]
        assert np.round(f2.jacobian, 2) == [1.33]

    def suite_arcsin():
        # arcsin() is undefined for |x| > 1
        with np.testing.assert_raises(ValueError):
            x = Var(3)
            np.arcsin(x)

        x = Var(1)
        f = np.arcsin(x)
        assert np.round(f.val, 2) == [1.57]
        np.testing.assert_array_equal(f.jacobian, np.array([np.nan]))

        x = Var(0)
        f = np.arcsin(x)
        assert f.val == [0.0]
        assert f.jacobian == [1.0]

    def suite_arccos():
        # arccos() is undefined for |x| > 1
        with np.testing.assert_raises(ValueError):
            x = Var(3)
            np.arccos(x)

        x = Var(0)
        f = np.arccos(x)
        assert np.round(f.val, 2) == [1.57]
        assert np.round(f.jacobian, 2) == [-1.0]

    def suite_arctan():
        x = Var(1)
        f = np.arctan(x)
        assert np.round(f.val, 2) == [0.79]
        assert np.round(f.jacobian, 2) == [0.5]

    def suite_sinh():
        x = Var(1)
        f = np.sinh(x)
        assert np.round(f.val, 2) == [1.18]
        assert np.round(f.jacobian, 2) == [1.54]

    def suite_cosh():
        x = Var(1)
        f = np.cosh(x)
        assert np.round(f.val, 2) == [1.54]
        assert np.round(f.jacobian, 2) == [1.18]

    def suite_tanh():
        x = Var(1)
        f = np.tanh(x)
        assert np.round(f.val, 2) == [0.76]
        assert np.round(f.jacobian, 2) == [0.42]

    def suite_sqrt():
        # derivative of 0^x does not exist if x < 1
        x = Var(0)
        with np.testing.assert_raises(ZeroDivisionError):
            f = np.sqrt(x)

        x1 = Var(9)
        f1 = np.sqrt(x1)
        assert f1 == Var(3, 1 / 6)

    def suite_log():
        # log() not defined for x <= 0
        with np.testing.assert_raises(ValueError):
            x0 = Var(0)
            f0 = x0.log(10)

        x1 = Var(1000)
        f1 = x1.log(10)
        assert np.round(f1.val, 2) == [3.0]
        assert np.round(f1.jacobian, 4) == [0.0004]

    def suite_exp():
        x = Var(5)
        f = np.exp(x)
        assert np.round(f.val, 2) == [148.41]
        assert np.round(f.jacobian, 2) == [148.41]

    def suite_logistic():
        x = Var(5)
        f = x.logistic()
        assert np.round(f.val, 4) == [0.9933]
        assert np.round(f.jacobian, 4) == [0.0066]

    suite_negative()
    suite_abs()
    suite_constant()
    suite_sin()
    suite_cos()
    suite_tan()
    suite_arcsin()
    suite_arccos()
    suite_arctan()
    suite_sinh()
    suite_cosh()
    suite_tanh()
    suite_sqrt()
    suite_log()
    suite_exp()
    suite_logistic()


def test_vector_input():

    def suite_negative():
        x = Var(4.0, [1, 0])
        y = Var(5.0, [0, 1])
        f = x ** 3 + 3 * y
        f1 = -f
        assert f1.val == np.array([-79.])
        np.testing.assert_array_equal(f1.jacobian, np.array([-48., -3., 0.]))

    def suite_abs():
        x = Var(-4.0, [1, 0])
        y = Var(-5.0, [0, 1])
        f = x ** 3 - 3 * y
        f1 = abs(f)
        np.testing.assert_array_equal(f1.val, np.array([49.]))
        np.testing.assert_array_equal(f1.jacobian, np.array([-48., -3., 0.]))

        # abs() not differentiable at zero
        with np.testing.assert_raises(ValueError):
            x = Var(0.0, [1, 0])
            y = Var(0.0, [0, 1])
            f = x ** 3 - 3 * y
            f1 = abs(f)

    def suite_constant():
        x = Var(4.0, None)
        y = Var(5.0, None)
        f = x ** 3 - 3 * y
        np.testing.assert_array_equal(f.val, np.array([49.]))
        np.testing.assert_array_equal(f.jacobian, np.array(None))

    def suite_sin():
        x = Var(3 * np.pi / 2, [1, 0])
        y = Var(np.pi / 2, [0, 1])
        f = 3 * np.sin(x) - 5 * np.sin(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([-8.]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([-0., -0.]))

    def suite_cos():
        x = Var(3 * np.pi / 2, [1, 0])
        y = Var(np.pi / 2, [0, 1])
        f = 3 * np.cos(x) - 5 * np.cos(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([-0.]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([3., 5.]))

    def suite_tan():
        x = Var(np.pi / 6, [1, 0])
        y = Var(np.pi / 4, [0, 1])
        f = 3 * np.tan(x) - 5 * np.tan(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([-3.27]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([4., -10.]))

        with np.testing.assert_raises(ValueError):
            z = Var(np.pi / 2)
            f = np.tan(z) - np.tan(x)

    def suite_arcsin():
        x = Var(1, [1, 0])
        y = Var(-1, [0, 1])
        f = np.arcsin(x) - 3 * np.arcsin(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([6.28]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([np.nan, np.nan]))

        x = Var(0.5, [1, 0])
        y = Var(0.2, [0, 1])
        f = np.arcsin(x) - 3 * np.arcsin(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([-0.08]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([1.15, -3.06]))

        # not defined for |x| > 1
        with np.testing.assert_raises(ValueError):
            x = Var(-1.01, [1, 0])
            f = 3 * np.arcsin(x) + 2 * np.arcsin(y)

    def suite_arccos():
        x = Var(0, [1, 0])
        y = Var(0.5, [0, 1])
        f = np.arccos(x) - 3 * np.arccos(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([-1.57]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([-1., 3.46]))

        # not defined for |x| > 1
        with np.testing.assert_raises(ValueError):
            x = Var(2, [1, 0])
            f = np.arccos(x) - np.arccos(y)

        x = Var(1, [1, 0])
        y = Var(-1, [0, 1])
        f = np.arccos(x) - 3 * np.arccos(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([-9.42]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([np.nan, np.nan]))

    def suite_arctan():
        x = Var(1, [1, 0])
        y = Var(- np.pi / 2, [0, 1])
        f = np.arctan(x) - 3 * np.arctan(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([3.8]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([0.5, -0.87]))

    def suite_sinh():
        x = Var(3, [1, 0])
        y = Var(1, [0, 1])
        f = np.sinh(x) - 3 * np.sinh(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([6.49]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([10.07, -4.63]))

    def suite_cosh():
        x = Var(3, [1, 0])
        y = Var(1, [0, 1])
        f = np.cosh(x) - 3 * np.cosh(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([5.44]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([10.02, -3.53]))

    def suite_tanh():
        x = Var(3, [1, 0])
        y = Var(1, [0, 1])
        f = np.tanh(x) - 3 * np.tanh(y)
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([-1.29]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([0.01, -1.26]))

    def suite_sqrt():
        x = Var(1.0, [1, 0, 0])
        y = Var(2.0, [0, 1, 0])
        z = Var(3.0, [0, 0, 1])
        f = 2 * x + y + z
        f1 = np.sqrt(f)
        np.testing.assert_array_equal(np.round(f1.val, 2), np.array([2.65]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([0.38, 0.19, 0.19]))

        # derivative of 0^x does not exist if x < 1
        with np.testing.assert_raises(ZeroDivisionError):
            f = np.sqrt(2 * x - y)

    def suite_log():
        x = Var(5.0, [1])
        f = Var([x + 2, x ** 3, 2 * x])
        f1 = f.log(2)
        np.testing.assert_array_equal(np.round(f1.val, 2), np.array([2.81, 6.97, 3.32]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([[0.21], [0.87], [0.29]]))

        # log() not defined for x <= 0
        with np.testing.assert_raises(ValueError):
            f2 = Var([x, -x, x ** 2])
            f3 = f2.log(2)

    def suite_exp():
        x = Var(1.0, [1, 0, 0])
        y = Var(2.0, [0, 1, 0])
        z = Var(3.0, [0, 0, 1])
        f = 2 * x - y + z
        f1 = np.exp(f)
        np.testing.assert_array_equal(np.round(f1.val, 2), np.array([20.09]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([40.17, -20.09, 20.09]))

    def suite_logistic():
        x = Var(1.0, [1, 0, 0])
        y = Var(2.0, [0, 1, 0])
        z = Var(3.0, [0, 0, 1])
        f = x + y + z
        f1 = f.logistic()
        np.testing.assert_array_equal(np.round(f1.val, 3), np.array([0.998]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 3), np.array([0.002, 0.002, 0.002]))

    suite_negative()
    suite_abs()
    suite_constant()
    suite_sin()
    suite_cos()
    suite_tan()
    suite_arcsin()
    suite_arccos()
    suite_arctan()
    suite_sinh()
    suite_cosh()
    suite_tanh()
    suite_sqrt()
    suite_log()
    suite_exp()
    suite_logistic()


def test_vector_input_1_to_n():

    def suite_negative():
        x = Var(2.0, [1])
        f = Var([x, 2 * x, x ** 2])
        f1 = -f
        np.testing.assert_array_equal(f1.val, np.array([-2., -4., -4.]))
        np.testing.assert_array_equal(f1.jacobian, np.array([[-1.], [-2.], [-4.]]))

    def suite_abs():
        # abs() not differentiable at zero
        with np.testing.assert_raises(ValueError):
            x = Var(0.0, [1, 0])
            f = Var([x, x ** 2, x ** 3])
            f1 = abs(f)

        x = Var(3.0, [1])
        f = Var([x, 2 * x, x ** 3])
        f1 = abs(f)
        np.testing.assert_array_equal(f1.val, np.array([3., 6., 27.]))
        np.testing.assert_array_equal(f1.jacobian, np.array([[1.], [2.], [27.]]))

    def suite_constant():
        x = Var(5.0)
        f = Var([x, x ** 2])
        np.testing.assert_array_equal(f.val, np.array([5., 25.]))
        np.testing.assert_array_equal(f.jacobian, np.array([[1.], [10.]]))

    def suite_sin():
        x = Var(np.pi / 2, [1])
        f = Var([np.sin(x) + 1, 3 * np.sin(x), np.sin(x) ** 3])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([2., 3., 1.]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[0.], [0.], [0.]]))

    def suite_cos():
        x = Var(np.pi / 2, [1])
        f = Var([np.cos(x) + 1, 3 * np.cos(x), np.cos(x) ** 3])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([1., 0., 0.]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[-1.], [-3.], [0.]]))

    def suite_tan():
        x = Var(np.pi / 4, [1])
        f = Var([np.tan(x) + 1, np.tan(x), np.tan(x) ** 2])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([2., 1., 1.]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[2.], [2.], [4.]]))

    def suite_arcsin():
        x = Var(0.5, [1])
        f = Var([2 * np.arcsin(x), np.arcsin(x) + 1, np.arcsin(x) ** 3])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([1.05, 1.52, 0.14]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[2.31], [1.15], [0.95]]))

        x = Var(1, [1])
        f = Var([2 * np.arcsin(x), np.arcsin(x) + 1, np.arcsin(x) ** 3])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([3.14, 2.57, 3.88]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[np.nan], [np.nan], [np.nan]]))

        # not defined for |x| > 1
        with np.testing.assert_raises(ValueError):
            z = Var(2, [1])
            f = Var([np.arcsin(z), np.arcsin(z) ** 2])

    def suite_arccos():
        x = Var(0.5, [1])
        f = Var([2 * np.arccos(x), np.arccos(x) + 1, np.arccos(x) ** 3])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([2.09, 2.05, 1.15]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[-2.31], [-1.15], [-3.8]]))

        x = Var(1, [1])
        f = Var([2 * np.arccos(x), np.arccos(x) + 1, np.arccos(x) ** 3])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([0., 1., 0.]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[np.nan], [np.nan], [np.nan]]))

        # not defined for |x| > 1
        with np.testing.assert_raises(ValueError):
            z = Var(2, [1])
            f = Var([np.arccos(z), np.arccos(z) ** 2])

    def suite_arctan():
        x = Var(1, [1])
        f = Var([np.arctan(x) ** 3, 2 * np.arctan(x)])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([0.48, 1.57]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[0.93], [1.]]))

    def suite_sinh():
        x = Var(1, [1])
        f = Var([np.sinh(x) ** 3, 2 * np.sinh(x)])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([1.62, 2.35]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[6.39], [3.09]]))

    def suite_cosh():
        x = Var(1, [1])
        f = Var([np.cosh(x) ** 3, 2 * np.cosh(x)])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([3.67, 3.09]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[8.39], [2.35]]))

    def suite_tanh():
        x = Var(1, [1])
        f = Var([np.tanh(x) ** 3, 2 * np.tanh(x)])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([0.44, 1.52]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[0.73], [0.84]]))


    def suite_sqrt():
        x = Var(2.0, [1])
        f = Var([x + 2, 3 * x, x ** 2])
        f1 = np.sqrt(f)
        np.testing.assert_array_equal(np.round(f1.val, 2), np.array([2., 2.45, 2.]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([[0.25], [0.61], [1.]]))

        # derivative of 0^x does not exist if x < 1
        with np.testing.assert_raises(ZeroDivisionError):
            x = Var(0.0, [1])
            f = Var([x, 3 * x])
            f = np.sqrt(f)


    def suite_log():
        x = Var(5.0, [1])
        f = Var([x + 2, x ** 3, 2 * x])
        f1 = f.log(2)
        np.testing.assert_array_equal(np.round(f1.val, 2), np.array([2.81, 6.97, 3.32]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([[0.21], [0.87], [0.29]]))

        # log() not defined for x <= 0
        with np.testing.assert_raises(ValueError):
            f2 = Var([x, -x, x ** 2])
            f3 = f2.log(2)


    def suite_exp():
        x = Var(2.0, [1])
        f = Var([x + 2, 3 * x, x ** 2])
        f1 = np.exp(f)
        np.testing.assert_array_equal(np.round(f1.val, 2), np.array([54.6, 403.43, 54.6]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([[54.6], [1210.29], [218.39]]))

    def suite_logistic():
        x = Var(2.0, [1])
        f = Var([x + 2, 3 * x, x ** 2])
        f1 = f.logistic()
        np.testing.assert_array_equal(np.round(f1.val, 3), np.array([0.982, 0.998, 0.982]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 3), np.array([[0.018], [0.007], [0.071]]))

    suite_negative()
    suite_abs()
    suite_constant()
    suite_sin()
    suite_cos()
    suite_tan()
    suite_arcsin()
    suite_arccos()
    suite_arctan()
    suite_sinh()
    suite_cosh()
    suite_tanh()
    suite_sqrt()
    suite_log()
    suite_exp()
    suite_logistic()


def suite__vector_input_m_to_n():

    def suite_neg():
        x = Var(1.0, [1, 0, 0])
        y = Var(2.0, [0, 1, 0])
        z = Var(3.0, [0, 0, 1])
        f = Var([x * y, y ** z, 3 * z])
        f1 = -f
        np.testing.assert_array_equal(f1.val, np.array([-2., -8., -9.]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([[-2., -1., 0.],
                                                                          [0., -12., -5.55],
                                                                          [0., 0., -3.]]))

    def suite_abs():
        x = Var(1.0, [1, 0, 0])
        y = Var(2.0, [0, 1, 0])
        z = Var(3.0, [0, 0, 1])
        f = Var([x * y, y ** z, 3 * z])
        f1 = abs(f)
        np.testing.assert_array_equal(f1.val, np.array([2., 8., 9.]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([[2., 1., 0.], [0., 12., 5.55], [0., 0., 3.]]))

    def suite_constant():
        x = Var(1.0)
        y = Var(2.0)
        z = Var(3.0)
        f = Var([x * y, y ** z, 3 * z])
        np.testing.assert_array_equal(f.val, np.array([2., 8., 9.]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[3.], [17.55], [3.]]))


    def suite_sin():
        x = Var(np.pi / 2, [1, 0, 0])
        y = Var(np.pi / 3, [0, 1, 0])
        z = Var(np.pi / 4, [0, 0, 1])
        f = Var([np.sin(x), 2 * np.sin(y), np.sin(z) ** 3])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([1., 1.73, 0.35]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[0., 0., 0.],
                                                                    [0., 1., 0.],
                                                                    [0., 0., 1.06]]))
    def suite_cos():
        x = Var(np.pi / 2, [1, 0, 0])
        y = Var(np.pi / 3, [0, 1, 0])
        z = Var(np.pi / 4, [0, 0, 1])
        f = Var([np.cos(x), 2 * np.cos(y), np.cos(z) ** 3])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([0., 1., 0.35]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[-1., 0., 0.],
                                                                    [0., -1.73, 0.],
                                                                    [0., 0., -1.06]]))

    def suite_tan():
        x = Var(np.pi / 3, [1, 0, 0])
        y = Var(np.pi / 4, [0, 1, 0])
        z = Var(np.pi / 6, [0, 0, 1])
        f = Var([np.tan(x), 2 * np.tan(y), np.tan(z) ** 3])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([1.73, 2., 0.19]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[4., 0., 0.],
                                                                    [0., 4., 0.],
                                                                    [0., 0., 1.33]]))
    def suite_arcsin():
        with np.testing.assert_raises(ValueError):
            x = Var(np.pi / 3, [1, 0, 0])
            y = Var(np.pi / 4, [0, 1, 0])
            z = Var(np.pi / 6, [0, 0, 1])
            f = Var([np.arcsin(x), 2 * np.arcsin(y), np.arcsin(z) ** 3 + 1])

        x = Var(0, [1, 0, 0])
        y = Var(1, [0, 1, 0])
        z = Var(-1, [0, 0, 1])
        f = Var([np.arcsin(x), 2 * np.arcsin(y), np.arcsin(z) ** 3 + 1])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([0., 3.14, -2.88]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[1., 0., 0.],
                                                                    [np.nan, np.nan, np.nan],
                                                                    [np.nan, np.nan, np.nan]]))
    def suite_arccos():
        with np.testing.assert_raises(ValueError):
            x = Var(np.pi / 3, [1, 0, 0])
            y = Var(np.pi / 4, [0, 1, 0])
            z = Var(np.pi / 6, [0, 0, 1])
            f = Var([np.arcsin(x), 2 * np.arcsin(y), np.arcsin(z) ** 3 + 1])

        x = Var(0, [1, 0, 0])
        y = Var(1, [0, 1, 0])
        z = Var(-1, [0, 0, 1])
        f = Var([np.arccos(x), 2 * np.arccos(y), np.arccos(z) ** 3 + 1])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([1.57, 0., 32.01]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[-1., 0., 0.],
                                                                    [np.nan, np.nan, np.nan],
                                                                    [np.nan, np.nan, np.nan]]))
    def suite_arctan():
        x = Var(np.pi / 3, [1, 0, 0])
        y = Var(np.pi / 4, [0, 1, 0])
        z = Var(np.pi / 6, [0, 0, 1])
        f = Var([np.arctan(x), 2 * np.arctan(y), np.arctan(y) ** 3 + 1])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([0.81, 1.33, 1.3]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[0.48, 0., 0.],
                                                                    [0., 1.24, 0.],
                                                                    [0., 0.82, 0.]]))
    def suite_sinh():
        x = Var(2, [1, 0])
        y = Var(3, [0, 1])
        f = Var([np.sinh(x) ** 2, 2 * np.sinh(y), np.sinh(x) * np.sinh(y)])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([13.15, 20.04, 36.33]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[27.29, 0.], [0., 20.14], [37.69, 36.51]]))

    def suite_cosh():
        x = Var(2, [1, 0])
        y = Var(3, [0, 1])
        f = Var([np.cosh(x) ** 2, 2 * np.cosh(y), np.cosh(x) * np.cosh(y)])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([14.15, 20.14, 37.88]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[27.29, 0.], [0., 20.04], [36.51, 37.69]]))

    def suite_tanh():
        x = Var(2, [1, 0])
        y = Var(3, [0, 1])
        f = Var([np.tanh(x) ** 2, 2 * np.tanh(y), np.tanh(x) * np.tanh(y)])
        np.testing.assert_array_equal(np.round(f.val, 2), np.array([0.93, 1.99, 0.96]))
        np.testing.assert_array_equal(np.round(f.jacobian, 2), np.array([[0.14, 0.], [0., 0.02], [0.07, 0.01]]))

    def suite_sqrt():
        x = Var(3.0, [1, 0, 0])
        y = Var(1.0, [0, 1, 0])
        f = Var([x ** 3, 2 * y, x * y])
        f1 = np.sqrt(f)
        np.testing.assert_array_equal(np.round(f1.val, 2), np.array([5.2, 1.41, 1.73]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([[2.6, 0., 0.],
                                                                          [0., 0.71, 0.],
                                                                          [0.29, 0.87, 0.]]))

        # derivative of 0^x does not exist if x < 1
        with np.testing.assert_raises(ZeroDivisionError):
            x = Var(0.0, [1])
            f = Var([x, 3 * x])
            f1 = np.sqrt(f)

    def suite_log():
        x = Var(3.0, [1, 0, 0])
        y = Var(1.0, [0, 1, 0])
        f = Var([x + 2, 3 * y, x ** y])
        f1 = f.log(10)
        np.testing.assert_array_equal(np.round(f1.val, 2), np.array([0.7, 0.48, 0.48]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([[0.09, 0., 0.], [0., 0.43, 0.], [0.14, 0.48, 0.]]))
        # not defined for |x| < 0
        with np.testing.assert_raises(ValueError):
            f2 = Var([-x, 3 * y, x ** y])
            f3 = f2.log(2)

    def suite_exp():
        x = Var(3.0, [1, 0, 0])
        y = Var(1.0, [0, 1, 0])
        f = Var([x + 2, 3 * y, x ** y])
        f1 = f.exp()
        np.testing.assert_array_equal(np.round(f1.val, 2), np.array([148.41, 20.09, 20.09]))
        np.testing.assert_array_equal(np.round(f1.jacobian, 2), np.array([[148.41, 0., 0.],
                                                                     [0., 60.26, 0.],
                                                                     [20.09, 66.2, 0.]]))

    def suite_logistic():
        x = Var(2.0, [1, 0, 0])
        y = Var(3.0, [0, 1, 0])
        f = Var([x + 2, 3 * y, x ** y])
        f1 = f.logistic()
        np.testing.assert_array_equal(np.around(f1.val, 3), np.array([0.982, 1., 1.]))
        np.testing.assert_array_equal(np.around(f1.jacobian, 3), np.array([[0.018, 0.000, 0.000],
                                                                      [0.000, 0.000, 0.000],
                                                                      [0.004, 0.002, 0.000]]))
    suite_neg()
    suite_abs()
    suite_constant()
    suite_sin()
    suite_cos()
    suite_tan()
    suite_arcsin()
    suite_arccos()
    suite_arctan()
    suite_sinh()
    suite_cosh()
    suite_tanh()
    suite_sqrt()
    suite_log()
    suite_exp()
    suite_logistic()




