from AutoDiff.ForwardAD import Var
import numpy as np
import math


def test_scalar_input():

    def suite_negative():
        x1 = Var(2.0)
        f1 = -x1
        assert f1.get_value() == -2.0
        assert f1.get_jacobian() == [-1.0]

        x2 = Var(0.0)
        f2 = -x2
        assert f2.get_value() == 0.0
        assert f2.get_jacobian() == [-1.0]

        # suite for operator order
        f3 = - x1 / x1
        assert f3.get_value() == -1.0
        assert f3.get_jacobian() == [0.0]

    def suite_abs():
        # abs() not differentiable at zero
        with np.testing.assert_raises(ValueError):
            x1 = Var(0.0)
            f1 = abs(x1)

        x2 = Var(5.0)
        f2 = abs(x2)
        assert f2.get_value() == 5.0
        assert f2.get_jacobian() == [1.0]

        x3 = Var(-5.0)
        f3 = abs(x3)
        assert f3.get_value() == 5.0
        assert f3.get_jacobian() == [-1.0]

    # def suite_constant():
    #     x = Var(4.0, None)
    #     f = x
    #     assert f.get_value() == 4.0
    #     assert f.get_jacobian() == None

    def suite_sin():
        x1 = Var(np.pi)
        f1 = 10e16 * Var.sin(x1)
        assert np.round(f1.get_value(), 2) == 12.25
        assert np.round(f1.get_jacobian(), 2) == [-1.e+17]

        x2 = Var(np.pi * 3 / 2)
        f2 = 10e16 * Var.sin(x2)
        assert np.round(f2.get_value(), 2) == -1.e+17
        assert np.round(f2.get_jacobian(), 2) == [-18.37]

    def suite_cos():
        x1 = Var(np.pi)
        f1 = 10e16 * Var.cos(x1)
        assert np.round(f1.get_value(), 2) == -1.e+17
        assert np.round(f1.get_jacobian(), 2) == [-12.25]

        x2 = Var(np.pi * 3 / 2)
        f2 = 10e16 * Var.cos(x2)
        assert np.round(f2.get_value(), 2) == -18.37
        assert np.round(f2.get_jacobian(), 2) == [1.e+17]

    def suite_tan():
        # tan() not define for multiples of pi/2
        with np.testing.assert_raises(ValueError):
            x0 = Var(np.pi / 2)
            f0 = Var.tan(x0)

        x1 = Var(np.pi / 3)
        f1 = Var.tan(x1)
        assert np.round(f1.get_value(), 2) == 1.73
        assert np.round(f1.get_jacobian(), 2) == [4.0]

        x2 = Var(np.pi / 6)
        f2 = Var.tan(x2)
        assert np.round(f2.get_value(), 2) == 0.58
        assert np.round(f2.get_jacobian(), 2) == [1.33]

    def suite_arcsin():
        # arcsin() is undefined for |x| > 1
        with np.testing.assert_raises(ValueError):
            x = Var(3)
            Var.arcsin(x)

        with np.testing.assert_raises(ZeroDivisionError):
            x = Var(1)
            f = Var.arcsin(x)
            # assert np.round(f.get_value(), 2) == 1.57
            # np.testing.assert_array_equal(f.get_jacobian(), np.array([np.nan]))

        x = Var(0)
        f = Var.arcsin(x)
        assert f.get_value() == [0.0]
        assert f.get_jacobian() == [1.0]

    def suite_arccos():
        # arccos() is undefined for |x| > 1
        with np.testing.assert_raises(ValueError):
            x = Var(3)
            Var.arccos(x)

        x = Var(0)
        f = Var.arccos(x)
        assert np.round(f.get_value(), 2) == 1.57
        assert np.round(f.get_jacobian(), 2) == [-1.0]

    def suite_arctan():
        x = Var(1)
        f = Var.arctan(x)
        assert np.round(f.get_value(), 2) == 0.79
        assert np.round(f.get_jacobian(), 2) == [0.5]

    def suite_sinh():
        x = Var(1)
        f = Var.sinh(x)
        assert np.round(f.get_value(), 2) == 1.18
        assert np.round(f.get_jacobian(), 2) == [1.54]

    def suite_cosh():
        x = Var(1)
        f = Var.cosh(x)
        assert np.round(f.get_value(), 2) == 1.54
        assert np.round(f.get_jacobian(), 2) == [1.18]

    def suite_tanh():
        x = Var(1)
        f = Var.tanh(x)
        assert np.round(f.get_value(), 2) == 0.76
        assert np.round(f.get_jacobian(), 2) == [0.42]

    def suite_sqrt():
        # derivative does not exist if x = 0
        x = Var(0)
        with np.testing.assert_raises(ZeroDivisionError):
            f = Var.sqrt(x)

        x1 = Var(9)
        f1 = Var.sqrt(x1)
        assert f1.get_value() == 3
        assert f1.get_jacobian == 1 / 6

    def suite_log():
        # log() not defined for x <= 0
        with np.testing.assert_raises(ValueError):
            x0 = Var(0)
            f0 = Var.log(x0)

        x1 = Var(1000)
        f1 = Var.log(x1)
        assert np.round(f1.get_value(), 2) == 3.0
        assert np.round(f1.get_jacobian(), 4) == [0.0004]

    def suite_exp():
        x = Var(5)
        f = Var.exp(x)
        assert np.round(f.get_value(), 2) == 148.41
        assert np.round(f.get_jacobian(), 2) == [148.41]

    def suite_logistic():
        x = Var(5)
        f = x.logistic()
        assert np.round(f.get_value(), 4) == 0.9933
        assert np.round(f.get_jacobian(), 4) == [0.0066]

    suite_negative()
    suite_abs()
    # suite_constant()
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

#
# def test_vector_input():
#
#     def suite_negative():
#         x = Var(4.0)
#         y = Var(5.0)
#         f = x ** 3 + 3 * y
#         f1 = -f
#         assert f1.get_value() == np.array([-79.])
#         np.testing.assert_array_equal(f1.get_jacobian(), np.array([-48., -3., 0.]))
#
#     def suite_abs():
#         x = Var(-4.0)
#         y = Var(-5.0)
#         f = x ** 3 - 3 * y
#         f1 = abs(f)
#         np.testing.assert_array_equal(f1.get_value(), np.array([49.]))
#         np.testing.assert_array_equal(f1.get_jacobian(), np.array([-48., -3., 0.]))
#
#         # abs() not differentiable at zero
#         with np.testing.assert_raises(ValueError):
#             x = Var(0.0)
#             y = Var(0.0)
#             f = x ** 3 - 3 * y
#             f1 = abs(f)
#
#     def suite_sin():
#         x = Var(3 * np.pi / 2)
#         y = Var(np.pi / 2)
#         f = 3 * Var.sin(x) - 5 * Var.sin(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([-8.]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([-0., -0.]))
#
#     def suite_cos():
#         x = Var(3 * np.pi / 2)
#         y = Var(np.pi / 2)
#         f = 3 * Var.cos(x) - 5 * Var.cos(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([-0.]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([3., 5.]))
#
#     def suite_tan():
#         x = Var(np.pi / 6)
#         y = Var(np.pi / 4)
#         f = 3 * Var.tan(x) - 5 * Var.tan(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([-3.27]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([4., -10.]))
#
#         with np.testing.assert_raises(ValueError):
#             z = Var(np.pi / 2)
#             f = Var.tan(z) - Var.tan(x)
#
#     def suite_arcsin():
#         x = Var(1)
#         y = Var(-1)
#         f = Var.arcsin(x) - 3 * Var.arcsin(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([6.28]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([np.nan, np.nan]))
#
#         x = Var(0.5)
#         y = Var(0.2)
#         f = Var.arcsin(x) - 3 * Var.arcsin(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([-0.08]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([1.15, -3.06]))
#
#         # not defined for |x| > 1
#         with np.testing.assert_raises(ValueError):
#             x = Var(-1.01)
#             f = 3 * Var.arcsin(x) + 2 * Var.arcsin(y)
#
#     def suite_arccos():
#         x = Var(0)
#         y = Var(0.5)
#         f = Var.arccos(x) - 3 * Var.arccos(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([-1.57]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([-1., 3.46]))
#
#         # not defined for |x| > 1
#         with np.testing.assert_raises(ValueError):
#             x = Var(2)
#             f = Var.arccos(x) - Var.arccos(y)
#
#         x = Var(1)
#         y = Var(-1)
#         f = Var.arccos(x) - 3 * Var.arccos(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([-9.42]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([np.nan, np.nan]))
#
#     def suite_arctan():
#         x = Var(1)
#         y = Var(- np.pi / 2)
#         f = Var.arctan(x) - 3 * Var.arctan(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([3.8]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([0.5, -0.87]))
#
#     def suite_sinh():
#         x = Var(3)
#         y = Var(1)
#         f = Var.sinh(x) - 3 * Var.sinh(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([6.49]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([10.07, -4.63]))
#
#     def suite_cosh():
#         x = Var(3)
#         y = Var(1)
#         f = Var.cosh(x) - 3 * Var.cosh(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([5.44]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([10.02, -3.53]))
#
#     def suite_tanh():
#         x = Var(3)
#         y = Var(1)
#         f = Var.tanh(x) - 3 * Var.tanh(y)
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([-1.29]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([0.01, -1.26]))
#
#     def suite_sqrt():
#         x = Var(1.0)
#         y = Var(2.0)
#         z = Var(3.0)
#         f = 2 * x + y + z
#         f1 = Var.sqrt(f)
#         np.testing.assert_array_equal(np.round(f1.get_value(), 2), np.array([2.65]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([0.38, 0.19, 0.19]))
#
#         # derivative of 0^x does not exist if x < 1
#         with np.testing.assert_raises(ZeroDivisionError):
#             f = Var.sqrt(2 * x - y)
#
#     def suite_log():
#         x = Var(5.0)
#         f = Var([x + 2, x ** 3, 2 * x])
#         f1 = f.log(2)
#         np.testing.assert_array_equal(np.round(f1.get_value(), 2), np.array([2.81, 6.97, 3.32]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([[0.21], [0.87], [0.29]]))
#
#         # log() not defined for x <= 0
#         with np.testing.assert_raises(ValueError):
#             f2 = Var([x, -x, x ** 2])
#             f3 = f2.log(2)
#
#     def suite_exp():
#         x = Var(1.0)
#         y = Var(2.0)
#         z = Var(3.0)
#         f = 2 * x - y + z
#         f1 = Var.exp(f)
#         np.testing.assert_array_equal(np.round(f1.get_value(), 2), np.array([20.09]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([40.17, -20.09, 20.09]))
#
#     def suite_logistic():
#         x = Var(1.0)
#         y = Var(2.0)
#         z = Var(3.0)
#         f = x + y + z
#         f1 = f.logistic()
#         np.testing.assert_array_equal(np.round(f1.get_value(), 3), np.array([0.998]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 3), np.array([0.002, 0.002, 0.002]))
#
#     suite_negative()
#     suite_abs()
#     suite_sin()
#     suite_cos()
#     suite_tan()
#     suite_arcsin()
#     suite_arccos()
#     suite_arctan()
#     suite_sinh()
#     suite_cosh()
#     suite_tanh()
#     suite_sqrt()
#     suite_log()
#     suite_exp()
#     suite_logistic()
#
#
# def test_vector_input_1_to_n():
#
#     def suite_negative():
#         x = Var(2.0)
#         f = Var([x, 2 * x, x ** 2])
#         f1 = -f
#         np.testing.assert_array_equal(f1.get_value(), np.array([-2., -4., -4.]))
#         np.testing.assert_array_equal(f1.get_jacobian(), np.array([[-1.], [-2.], [-4.]]))
#
#     def suite_abs():
#         # abs() not differentiable at zero
#         with np.testing.assert_raises(ValueError):
#             x = Var(0.0)
#             f = Var([x, x ** 2, x ** 3])
#             f1 = abs(f)
#
#         x = Var(3.0)
#         f = Var([x, 2 * x, x ** 3])
#         f1 = abs(f)
#         np.testing.assert_array_equal(f1.get_value(), np.array([3., 6., 27.]))
#         np.testing.assert_array_equal(f1.get_jacobian(), np.array([[1.], [2.], [27.]]))
#
#     # def suite_constant():
#     #     x = Var(5.0)
#     #     f = Var([x, x ** 2])
#     #     np.testing.assert_array_equal(f.get_value(), np.array([5., 25.]))
#     #     np.testing.assert_array_equal(f.get_jacobian(), np.array([[1.], [10.]]))
#
#     def suite_sin():
#         x = Var(np.pi / 2)
#         f = Var([Var.sin(x) + 1, 3 * Var.sin(x), Var.sin(x) ** 3])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([2., 3., 1.]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[0.], [0.], [0.]]))
#
#     def suite_cos():
#         x = Var(np.pi / 2)
#         f = Var([Var.cos(x) + 1, 3 * Var.cos(x), Var.cos(x) ** 3])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([1., 0., 0.]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[-1.], [-3.], [0.]]))
#
#     def suite_tan():
#         x = Var(np.pi / 4)
#         f = Var([Var.tan(x) + 1, Var.tan(x), Var.tan(x) ** 2])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([2., 1., 1.]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[2.], [2.], [4.]]))
#
#     def suite_arcsin():
#         x = Var(0.5)
#         f = Var([2 * Var.arcsin(x), Var.arcsin(x) + 1, Var.arcsin(x) ** 3])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([1.05, 1.52, 0.14]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[2.31], [1.15], [0.95]]))
#
#         x = Var(1)
#         f = Var([2 * Var.arcsin(x), Var.arcsin(x) + 1, Var.arcsin(x) ** 3])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([3.14, 2.57, 3.88]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[np.nan], [np.nan], [np.nan]]))
#
#         # not defined for |x| > 1
#         with np.testing.assert_raises(ValueError):
#             z = Var(2)
#             f = Var([Var.arcsin(z), Var.arcsin(z) ** 2])
#
#     def suite_arccos():
#         x = Var(0.5)
#         f = Var([2 * Var.arccos(x), Var.arccos(x) + 1, Var.arccos(x) ** 3])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([2.09, 2.05, 1.15]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[-2.31], [-1.15], [-3.8]]))
#
#         x = Var(1)
#         f = Var([2 * Var.arccos(x), Var.arccos(x) + 1, Var.arccos(x) ** 3])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([0., 1., 0.]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[np.nan], [np.nan], [np.nan]]))
#
#         # not defined for |x| > 1
#         with np.testing.assert_raises(ValueError):
#             z = Var(2)
#             f = Var([Var.arccos(z), Var.arccos(z) ** 2])
#
#     def suite_arctan():
#         x = Var(1)
#         f = Var([Var.arctan(x) ** 3, 2 * Var.arctan(x)])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([0.48, 1.57]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[0.93], [1.]]))
#
#     def suite_sinh():
#         x = Var(1)
#         f = Var([Var.sinh(x) ** 3, 2 * Var.sinh(x)])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([1.62, 2.35]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[6.39], [3.09]]))
#
#     def suite_cosh():
#         x = Var(1)
#         f = Var([Var.cosh(x) ** 3, 2 * Var.cosh(x)])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([3.67, 3.09]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[8.39], [2.35]]))
#
#     def suite_tanh():
#         x = Var(1)
#         f = Var([Var.tanh(x) ** 3, 2 * Var.tanh(x)])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([0.44, 1.52]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[0.73], [0.84]]))
#
#
#     def suite_sqrt():
#         x = Var(2.0)
#         f = Var([x + 2, 3 * x, x ** 2])
#         f1 = Var.sqrt(f)
#         np.testing.assert_array_equal(np.round(f1.get_value(), 2), np.array([2., 2.45, 2.]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([[0.25], [0.61], [1.]]))
#
#         # derivative of 0^x does not exist if x < 1
#         with np.testing.assert_raises(ZeroDivisionError):
#             x = Var(0.0)
#             f = Var([x, 3 * x])
#             f = Var.sqrt(f)
#
#
#     def suite_log():
#         x = Var(5.0)
#         f = Var([x + 2, x ** 3, 2 * x])
#         f1 = f.log(2)
#         np.testing.assert_array_equal(np.round(f1.get_value(), 2), np.array([2.81, 6.97, 3.32]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([[0.21], [0.87], [0.29]]))
#
#         # log() not defined for x <= 0
#         with np.testing.assert_raises(ValueError):
#             f2 = Var([x, -x, x ** 2])
#             f3 = f2.log(2)
#
#
#     def suite_exp():
#         x = Var(2.0)
#         f = Var([x + 2, 3 * x, x ** 2])
#         f1 = Var.exp(f)
#         np.testing.assert_array_equal(np.round(f1.get_value(), 2), np.array([54.6, 403.43, 54.6]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([[54.6], [1210.29], [218.39]]))
#
#     def suite_logistic():
#         x = Var(2.0)
#         f = Var([x + 2, 3 * x, x ** 2])
#         f1 = f.logistic()
#         np.testing.assert_array_equal(np.round(f1.get_value(), 3), np.array([0.982, 0.998, 0.982]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 3), np.array([[0.018], [0.007], [0.071]]))
#
#     suite_negative()
#     suite_abs()
#     # suite_constant()
#     suite_sin()
#     suite_cos()
#     suite_tan()
#     suite_arcsin()
#     suite_arccos()
#     suite_arctan()
#     suite_sinh()
#     suite_cosh()
#     suite_tanh()
#     suite_sqrt()
#     suite_log()
#     suite_exp()
#     suite_logistic()
#
#
# def suite__vector_input_m_to_n():
#
#     def suite_neg():
#         x = Var(1.0)
#         y = Var(2.0)
#         z = Var(3.0)
#         f = Var([x * y, y ** z, 3 * z])
#         f1 = -f
#         np.testing.assert_array_equal(f1.get_value(), np.array([-2., -8., -9.]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([[-2., -1., 0.],
#                                                                           [0., -12., -5.55],
#                                                                           [0., 0., -3.]]))
#
#     def suite_abs():
#         x = Var(1.0)
#         y = Var(2.0)
#         z = Var(3.0)
#         f = Var([x * y, y ** z, 3 * z])
#         f1 = abs(f)
#         np.testing.assert_array_equal(f1.get_value(), np.array([2., 8., 9.]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([[2., 1., 0.], [0., 12., 5.55], [0., 0., 3.]]))
#
#     # def suite_constant():
#     #     x = Var(1.0)
#     #     y = Var(2.0)
#     #     z = Var(3.0)
#     #     f = Var([x * y, y ** z, 3 * z])
#     #     np.testing.assert_array_equal(f.get_value(), np.array([2., 8., 9.]))
#     #     np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[3.], [17.55], [3.]]))
#
#
#     def suite_sin():
#         x = Var(np.pi / 2)
#         y = Var(np.pi / 3)
#         z = Var(np.pi / 4)
#         f = Var([Var.sin(x), 2 * Var.sin(y), Var.sin(z) ** 3])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([1., 1.73, 0.35]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[0., 0., 0.],
#                                                                     [0., 1., 0.],
#                                                                     [0., 0., 1.06]]))
#     def suite_cos():
#         x = Var(np.pi / 2)
#         y = Var(np.pi / 3)
#         z = Var(np.pi / 4)
#         f = Var([Var.cos(x), 2 * Var.cos(y), Var.cos(z) ** 3])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([0., 1., 0.35]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[-1., 0., 0.],
#                                                                     [0., -1.73, 0.],
#                                                                     [0., 0., -1.06]]))
#
#     def suite_tan():
#         x = Var(np.pi / 3)
#         y = Var(np.pi / 4)
#         z = Var(np.pi / 6)
#         f = Var([Var.tan(x), 2 * Var.tan(y), Var.tan(z) ** 3])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([1.73, 2., 0.19]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[4., 0., 0.],
#                                                                     [0., 4., 0.],
#                                                                     [0., 0., 1.33]]))
#     def suite_arcsin():
#         with np.testing.assert_raises(ValueError):
#             x = Var(np.pi / 3)
#             y = Var(np.pi / 4)
#             z = Var(np.pi / 6)
#             f = Var([Var.arcsin(x), 2 * Var.arcsin(y), Var.arcsin(z) ** 3 + 1])
#
#         x = Var(0)
#         y = Var(1)
#         z = Var(-1)
#         f = Var([Var.arcsin(x), 2 * Var.arcsin(y), Var.arcsin(z) ** 3 + 1])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([0., 3.14, -2.88]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[1., 0., 0.],
#                                                                     [np.nan, np.nan, np.nan],
#                                                                     [np.nan, np.nan, np.nan]]))
#     def suite_arccos():
#         with np.testing.assert_raises(ValueError):
#             x = Var(np.pi / 3)
#             y = Var(np.pi / 4)
#             z = Var(np.pi / 6)
#             f = Var([Var.arcsin(x), 2 * Var.arcsin(y), Var.arcsin(z) ** 3 + 1])
#
#         x = Var(0)
#         y = Var(1)
#         z = Var(-1)
#         f = Var([Var.arccos(x), 2 * Var.arccos(y), Var.arccos(z) ** 3 + 1])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([1.57, 0., 32.01]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[-1., 0., 0.],
#                                                                     [np.nan, np.nan, np.nan],
#                                                                     [np.nan, np.nan, np.nan]]))
#     def suite_arctan():
#         x = Var(np.pi / 3)
#         y = Var(np.pi / 4)
#         z = Var(np.pi / 6)
#         f = Var([Var.arctan(x), 2 * Var.arctan(y), Var.arctan(y) ** 3 + 1])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([0.81, 1.33, 1.3]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[0.48, 0., 0.],
#                                                                     [0., 1.24, 0.],
#                                                                     [0., 0.82, 0.]]))
#     def suite_sinh():
#         x = Var(2)
#         y = Var(3)
#         f = Var([Var.sinh(x) ** 2, 2 * Var.sinh(y), Var.sinh(x) * Var.sinh(y)])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([13.15, 20.04, 36.33]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[27.29, 0.], [0., 20.14], [37.69, 36.51]]))
#
#     def suite_cosh():
#         x = Var(2)
#         y = Var(3)
#         f = Var([Var.cosh(x) ** 2, 2 * Var.cosh(y), Var.cosh(x) * Var.cosh(y)])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([14.15, 20.14, 37.88]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[27.29, 0.], [0., 20.04], [36.51, 37.69]]))
#
#     def suite_tanh():
#         x = Var(2)
#         y = Var(3)
#         f = Var([Var.tanh(x) ** 2, 2 * Var.tanh(y), Var.tanh(x) * Var.tanh(y)])
#         np.testing.assert_array_equal(np.round(f.get_value(), 2), np.array([0.93, 1.99, 0.96]))
#         np.testing.assert_array_equal(np.round(f.get_jacobian(), 2), np.array([[0.14, 0.], [0., 0.02], [0.07, 0.01]]))
#
#     def suite_sqrt():
#         x = Var(3.0)
#         y = Var(1.0)
#         f = Var([x ** 3, 2 * y, x * y])
#         f1 = Var.sqrt(f)
#         np.testing.assert_array_equal(np.round(f1.get_value(), 2), np.array([5.2, 1.41, 1.73]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([[2.6, 0., 0.],
#                                                                           [0., 0.71, 0.],
#                                                                           [0.29, 0.87, 0.]]))
#
#         # derivative of 0^x does not exist if x < 1
#         with np.testing.assert_raises(ZeroDivisionError):
#             x = Var(0.0)
#             f = Var([x, 3 * x])
#             f1 = Var.sqrt(f)
#
#     def suite_log():
#         x = Var(3.0)
#         y = Var(1.0)
#         f = Var([x + 2, 3 * y, x ** y])
#         f1 = f.log(10)
#         np.testing.assert_array_equal(np.round(f1.get_value(), 2), np.array([0.7, 0.48, 0.48]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([[0.09, 0., 0.], [0., 0.43, 0.], [0.14, 0.48, 0.]]))
#         # not defined for |x| < 0
#         with np.testing.assert_raises(ValueError):
#             f2 = Var([-x, 3 * y, x ** y])
#             f3 = f2.log(2)
#
#     def suite_exp():
#         x = Var(3.0)
#         y = Var(1.0)
#         f = Var([x + 2, 3 * y, x ** y])
#         f1 = f.exp()
#         np.testing.assert_array_equal(np.round(f1.get_value(), 2), np.array([148.41, 20.09, 20.09]))
#         np.testing.assert_array_equal(np.round(f1.get_jacobian(), 2), np.array([[148.41, 0., 0.],
#                                                                      [0., 60.26, 0.],
#                                                                      [20.09, 66.2, 0.]]))
#
#     def suite_logistic():
#         x = Var(2.0)
#         y = Var(3.0)
#         f = Var([x + 2, 3 * y, x ** y])
#         f1 = f.logistic()
#         np.testing.assert_array_equal(Var.around(f1.get_value(), 3), np.array([0.982, 1., 1.]))
#         np.testing.assert_array_equal(Var.around(f1.get_jacobian(), 3), np.array([[0.018, 0.000, 0.000],
#                                                                       [0.000, 0.000, 0.000],
#                                                                       [0.004, 0.002, 0.000]]))
#     suite_neg()
#     suite_abs()
#     # suite_constant()
#     suite_sin()
#     suite_cos()
#     suite_tan()
#     suite_arcsin()
#     suite_arccos()
#     suite_arctan()
#     suite_sinh()
#     suite_cosh()
#     suite_tanh()
#     suite_sqrt()
#     suite_log()
#     suite_exp()
#     suite_logistic()
#
#
