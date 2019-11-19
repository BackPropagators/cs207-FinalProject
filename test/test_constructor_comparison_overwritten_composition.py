from AutoDiff.ForwardAD import Var
import numpy as np
import math

def test_constructor():
    def test_np_input():
        x = Var(1.0)
        assert x.get_value() == 1.0
        assert type(x.get_value()) == float
        assert x.get_jacobian() == [1.0]

    test_np_input()


def overwritten_elementary():
    def test_scalar_input():
        x = Var(4.0)
        def suite_add():
            z2 = x+2+x+10
            assert z2.get_value() == 20.0
            assert z2.get_jacobian() == [2.0]

            # x not modified
            assert x.get_value() == Var(4.0).get_value()

        def suite_radd():
            z1 = 2+x+10+x
            assert z1.get_value() == 20.0
            assert z1.get_jacobian() == [2.0]

            z2 = 1.0+(3.0+2.0)+x+x
            assert z2.get_value() == 14.0
            assert z1.get_jacobian() == [2.0]

        def suite_subtract():
            x = Var(4.0)

            z2 = x -1.0 -2.0 -x
            assert z2.get_value() == -3.0
            assert z2.get_jacobian() == [0]

            # x not modified
            assert x.get_value() == Var(4.0).get_value()

        def suite_rsubtract():
            x = Var(4.0)
            z1 = 10.0-x
            assert z1.get_value() == 6.0
            assert z1.get_jacobian() == [-1.0]

            z2 = 20.0-3.0-x-x
            assert z2.get_value() == 9.0
            assert z2.get_jacobian() == [-2.0]

        def suite_mul():
            x = Var(4.0)

            z2 = x*2*3
            assert z2.get_value() == 24.0
            assert z2.get_jacobian() == [6.0]

            z3 = x*x+x*2
            assert z3.get_value() == 24.0
            assert z3.get_jacobian() == [10.0]

        def suite_rmul():
            z1 = 3*x
            assert z1.get_value()  == 12.0
            assert z1.get_jacobian() == [3.0]

            z2 = 3*10*x*x
            assert z2.get_value() == 480.0
            assert z2.get_jacobian() == [240.0]

        def suite_div():
            z2 = x/4
            assert z2.get_value() == 1.0
            assert z2.get_jacobian() == [0.25]

            z3 = (x/0.5)/0.1
            assert z3.get_value() == 80.0
            assert z3.get_jacobian() == [20.0]

        def suite_rdiv():
            z1 = 8.0/x
            assert z1.get_value() == 2.0
            assert z1.get_jacobian() == [-0.5]

            z2 = (24.0/x)/1.5
            assert z2.get_value() == 4.0
            assert z2.get_jacobian() == [-1.0]

        def suite_pow():
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

        def suite_rpow():
            a = 2
            z1 = a**x
            assert z1.get_value() == 16.0
            assert z1.get_jacobian() == [np.log(a)*16.0]

        suite_add()
        suite_radd()
        suite_subtract()
        suite_rsubtract()
        suite_mul()
        suite_rmul()
        suite_div()
        suite_rdiv()
        suite_pow()
        suite_rpow()

    test_scalar_input()


def composition():
    x = Var(np.pi)

    def trig():
        z1 = np.sin(np.cos(x))
        assert np.round(z1.get_value(),10) == -0.8414709848
        assert np.round(z1.get_jacobian(),10) == [0]
    trig()


test_constructor()
overwritten_elementary()
composition()
