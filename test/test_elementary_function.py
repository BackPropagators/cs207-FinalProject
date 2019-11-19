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
        assert np.round(f1.get_jacobian(), 2) == [0.17]

    def suite_log():
        # log() not defined for x <= 0
        with np.testing.assert_raises(ValueError):
            x0 = Var(0)
            f0 = Var.log(x0, 10)

        x1 = Var(1000)
        f1 = Var.log(x1, 10)
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

test_scalar_input()
