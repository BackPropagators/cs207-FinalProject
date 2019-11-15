import AutoDiff.ForwardAd as Var
import numpy as np
import math


def test_scalar_input():

    def suite_negative():
        x1 = Var(np.array([2.0]))
        f1 = -x1
        assert f.val == [-2.0]
        assert f.jacobian == [-1.0]

        x2 = Var(np.array([0.0]))
        f2 = -x2
        assert f2.val == [0.0]
        assert f2.jacobian == [-1.0]

        # test for operator order
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
        assert f1 == Var(3, 1/6)

    def suite_log():

        # log() not defined for x <= 0
        with np.testing.assert_raises(ValueError):
            x0 = Var(0)
            f0 = x2.log(10)

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