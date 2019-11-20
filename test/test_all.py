from AutoDiff.ForwardAD import Var
import numpy as np

def test_constructor():
    x = Var(1.0)
    assert x.get_value() == 1.0
    assert type(x.get_value()) == float
    assert x.get_jacobian() == [1.0]


def test_overloading():
    def test_scalar_input():
        x = Var(4.0)
        def test_add():
            z2 = x+2+x+10
            assert z2.get_value() == 20.0
            assert z2.get_jacobian() == [2.0]

            # x not modified
            assert x.get_value() == Var(4.0).get_value()

        def test_radd():
            z1 = 2+x+10+x
            assert z1.get_value() == 20.0
            assert z1.get_jacobian() == [2.0]

            z2 = 1.0+(3.0+2.0)+x+x
            assert z2.get_value() == 14.0
            assert z1.get_jacobian() == [2.0]

        def test_subtract():
            x = Var(4.0)

            z2 = x -1.0 -2.0 -x
            assert z2.get_value() == -3.0
            assert z2.get_jacobian() == [0]

            # x not modified
            assert x.get_value() == Var(4.0).get_value()

        def test_rsubtract():
            x = Var(4.0)
            z1 = 10.0-x
            assert z1.get_value() == 6.0
            assert z1.get_jacobian() == [-1.0]

            z2 = 20.0-3.0-x-x
            assert z2.get_value() == 9.0
            assert z2.get_jacobian() == [-2.0]

        def test_mul():
            x = Var(4.0)

            z2 = x*2*3
            assert z2.get_value() == 24.0
            assert z2.get_jacobian() == [6.0]

            z3 = x*x+x*2
            assert z3.get_value() == 24.0
            assert z3.get_jacobian() == [10.0]

        def test_rmul():
            z1 = 3*x
            assert z1.get_value()  == 12.0
            assert z1.get_jacobian() == [3.0]

            z2 = 3*10*x*x
            assert z2.get_value() == 480.0
            assert z2.get_jacobian() == [240.0]

        def test_div():
            z2 = x/4
            assert z2.get_value() == 1.0
            assert z2.get_jacobian() == [0.25]

            z3 = (x/0.5)/0.1
            assert z3.get_value() == 80.0
            assert z3.get_jacobian() == [20.0]

        def test_rdiv():
            z1 = 8.0/x
            assert z1.get_value() == 2.0
            assert z1.get_jacobian() == [-0.5]

            z2 = (24.0/x)/1.5
            assert z2.get_value() == 4.0
            assert z2.get_jacobian() == [-1.0]

        def test_pow():
            z1 = x**2
            z3 = x*x
            assert z1.get_value() == 16.0
            assert z1.get_jacobian() == [8.0]
            assert z1.get_value() == z3.get_value()

            z2 = x**(0.5)
            z3 = Var.sqrt(x)
            z4 = x.sqrt()
            assert z2.get_value() == 2.0
            assert z2.get_jacobian() == [0.25]
            assert z2.get_value() == z3.get_value() == z4.get_value()

        def test_rpow():
            a = 2
            z1 = a**x
            assert z1.get_value() == 16.0
            assert z1.get_jacobian() == [np.log(a)*16.0]

        test_add()
        test_radd()
        test_subtract()
        test_rsubtract()
        test_mul()
        test_rmul()
        test_div()
        test_rdiv()
        test_pow()
        test_rpow()

    test_scalar_input()


def test_elementary():

    def test_negative():
        x1 = Var(2.0)
        f1 = -x1
        assert f1.get_value() == -2.0
        assert f1.get_jacobian() == [-1.0]

        x2 = Var(0.0)
        f2 = -x2
        assert f2.get_value() == 0.0
        assert f2.get_jacobian() == [-1.0]

        # test for operator order
        f3 = - x1 / x1
        assert f3.get_value() == -1.0
        assert f3.get_jacobian() == [0.0]

    def test_abs():
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

    def test_sin():
        x1 = Var(np.pi)
        f1 = 10e16 * Var.sin(x1)
        assert np.round(f1.get_value(), 2) == 12.25
        assert np.round(f1.get_jacobian(), 2) == [-1.e+17]

        x2 = Var(np.pi * 3 / 2)
        f2 = 10e16 * Var.sin(x2)
        assert np.round(f2.get_value(), 2) == -1.e+17
        assert np.round(f2.get_jacobian(), 2) == [-18.37]

    def test_cos():
        x1 = Var(np.pi)
        f1 = 10e16 * Var.cos(x1)
        assert np.round(f1.get_value(), 2) == -1.e+17
        assert np.round(f1.get_jacobian(), 2) == [-12.25]

        x2 = Var(np.pi * 3 / 2)
        f2 = 10e16 * Var.cos(x2)
        assert np.round(f2.get_value(), 2) == -18.37
        assert np.round(f2.get_jacobian(), 2) == [1.e+17]

    def test_tan():
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

    def test_arcsin():
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

    def test_arccos():
        # arccos() is undefined for |x| > 1
        with np.testing.assert_raises(ValueError):
            x = Var(3)
            Var.arccos(x)

        x = Var(0)
        f = Var.arccos(x)
        assert np.round(f.get_value(), 2) == 1.57
        assert np.round(f.get_jacobian(), 2) == [-1.0]

    def test_arctan():
        x = Var(1)
        f = Var.arctan(x)
        assert np.round(f.get_value(), 2) == 0.79
        assert np.round(f.get_jacobian(), 2) == [0.5]

    def test_sinh():
        x = Var(1)
        f = Var.sinh(x)
        assert np.round(f.get_value(), 2) == 1.18
        assert np.round(f.get_jacobian(), 2) == [1.54]

    def test_cosh():
        x = Var(1)
        f = Var.cosh(x)
        assert np.round(f.get_value(), 2) == 1.54
        assert np.round(f.get_jacobian(), 2) == [1.18]

    def test_tanh():
        x = Var(1)
        f = Var.tanh(x)
        assert np.round(f.get_value(), 2) == 0.76
        assert np.round(f.get_jacobian(), 2) == [0.42]

    def test_sqrt():
        # derivative does not exist if x = 0
        x = Var(0)
        with np.testing.assert_raises(ZeroDivisionError):
            f = Var.sqrt(x)

        x1 = Var(9)
        f1 = Var.sqrt(x1)
        assert f1.get_value() == 3
        assert np.round(f1.get_jacobian(), 2) == [0.17]

    def test_log():
        # log() not defined for x <= 0
        with np.testing.assert_raises(ValueError):
            x0 = Var(0)
            f0 = Var.log(x0, 10)

        x1 = Var(1000)
        f1 = Var.log(x1, 10)
        assert np.round(f1.get_value(), 2) == 3.0
        assert np.round(f1.get_jacobian(), 4) == [0.0004]

    def test_exp():
        x = Var(5)
        f = Var.exp(x)
        assert np.round(f.get_value(), 2) == 148.41
        assert np.round(f.get_jacobian(), 2) == [148.41]

    test_negative()
    test_abs()
    test_sin()
    test_cos()
    test_tan()
    test_arcsin()
    test_arccos()
    test_arctan()
    test_sinh()
    test_cosh()
    test_tanh()
    test_sqrt()
    test_log()
    test_exp()


def test_composition():
    x = Var(np.pi)

    def test_1_trig():
        z1 = Var.sin(Var.cos(x))
        assert np.round(z1.get_value(),10) == -0.8414709848
        assert np.round(z1.get_jacobian(),10) == [0]

    def test_2():
        z2 = Var.sin(x**2)
        assert np.round(z2.get_value(), 9) == -0.430301217
        assert np.round(z2.get_jacobian(),10) == -5.6717394031
    test_1_trig()
    test_2()


test_constructor()
test_overloading()
test_elementary()
test_composition()
