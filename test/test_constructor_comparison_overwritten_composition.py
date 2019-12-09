from AutoDiff.ForwardAD import Var
from AutoDiff.ForwardAD import MultiFunc
import numpy as np
import math

#import sys
#sys.path.append('AutoDiff/ForwardAD')
#from ForwardAD import Var, MultiFunc

## test of comparison
## get_derivative_of in MultiFunc
## 439line: test_div()

#input is an array, output is a list
def test_constructor():
    x = Var(1.0)
    y = Var(2.0)   
    # scalar input and scalar output
    def test_scalar_scalar():
        assert x.get_value() == 1.0
        assert type(x.get_value()) == float
        assert x.get_der() == [1.0]
        
    # vector input and scalar output    
    def test_vec_scalar():
        z = 2*x+y**2
        assert z.get_value() == 6.0
        assert type(z.get_value()) == float
        assert z._get_derivative_of(x) == 2.0
        assert z._get_derivative_of(y) == 4.0
        assert type(z._get_derivative_of(x)) == float
        assert type(z._get_derivative_of(y)) == float
        assert z.get_der([x,y]) == [2.0, 4.0]
        assert type(z.get_der([x,y])) == list
        
    # scalar input and vector output
    def test_scalar_vector():
        vec = MultiFunc([x, x**2]) 
        assert vec.get_value() == [1.0, 1.0]
        assert vec.get_der([x]) == [[1.0], [2.0]]
        assert type(vec.get_value()) == list
        assert type(vec.get_der([x])) == list
        
    # vector input and vector output
    def test_vector_vector():
        z = MultiFunc([x+y, y**2])
        assert z.get_value() == [3.0, 4.0]
        assert z.get_der([x,y]) == [[1.0, 1.0], [0, 4.0]]
        ## partial derivative
        assert type(z.get_value()) == list
        assert type(z.get_der([x,y])) == list

    test_scalar_scalar()
    test_vec_scalar()
    test_scalar_vector()
    test_vector_vector()
    
def test_scalar_scalar():
    def test_overloading():
        x = Var(4.0)
        #y = Var(3.0)
        def test_add():
            # scalar->scalar   
            z1 = x+2+x+10
            assert z1.get_value() == 20.0
            assert z1.get_der() == [2.0]
            # x not modified
            assert x.get_value() == Var(4.0).get_value()

        def test_radd():
            z1 = 2+x+10+x
            assert z1.get_value() == 20.0
            assert z1.get_der() == [2.0]

            z2 = 1.0+(3.0+2.0)+x+x
            assert z2.get_value() == 14.0
            assert z1.get_der() == [2.0]

        def test_subtract():
            x = Var(4.0)

            z2 = x -1.0 -2.0 -x
            assert z2.get_value() == -3.0
            assert z2.get_der() == [0]

            # x not modified
            assert x.get_value() == Var(4.0).get_value()

        def test_rsubtract():
            x = Var(4.0)
            z1 = 10.0-x
            assert z1.get_value() == 6.0
            assert z1.get_der() == [-1.0]

            z2 = 20.0-3.0-x-x
            assert z2.get_value() == 9.0
            assert z2.get_der() == [-2.0]

        def test_mul():
            x = Var(4.0)

            z2 = x*2*3
            assert z2.get_value() == 24.0
            assert z2.get_der() == [6.0]

            z3 = x*x+x*2
            assert z3.get_value() == 24.0
            assert z3.get_der() == [10.0]

        def test_rmul():
            z1 = 3*x
            assert z1.get_value()  == 12.0
            assert z1.get_der() == [3.0]

            z2 = 3*10*x*x
            assert z2.get_value() == 480.0
            assert z2.get_der() == [240.0]

        def test_div():
            z2 = x/4
            assert z2.get_value() == 1.0
            assert z2.get_der() == [0.25]

            z3 = (x/0.5)/0.1
            assert z3.get_value() == 80.0
            assert np.round(z3.get_der(), 10) == np.round([20.0], 10)

        def test_rdiv():
            z1 = 8.0/x
            assert z1.get_value() == 2.0
            assert z1.get_der() == [-0.5]

            z2 = (24.0/x)/1.5
            assert z2.get_value() == 4.0
            assert z2.get_der() == [-1.0]

        def test_pow():
            z1 = x**2
            z3 = x*x
            assert z1.get_value() == 16.0
            assert z1.get_der() == [8.0]
            assert z1.get_value() == z3.get_value()

            z2 = x**(0.5)
            z3 = Var.sqrt(x)
            z4 = x.sqrt()
            assert z2.get_value() == 2.0
            assert z2.get_der() == [0.25]
            assert z2.get_value() == z3.get_value() == z4.get_value()

        def test_rpow():
            a = 2
            z1 = a**x
            assert z1.get_value() == 16.0
            assert z1.get_der() == [np.log(a)*16.0]

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



    
    def test_comparison():
        x = Var(4.0)
        y = Var(1.5)
        q = Var(4.0)
        m1 = x + x
        m2 = 2*x
        z1 = x*4.0/2.0
        z2 = x*2.0
            
        def test_eq():
            assert not x == y
            assert not x == q
            
            assert m1 == m2
            assert m1.__eq__(m2)
            
            assert z1 == z2
            
        def test_ne():
            z3 = -z1
            assert not z3 == z2
            
            m3 = -m1
            m3.get_value() == -8.0
            
        '''    
        def test_less():
            assert x > y
            assert not z < y
            assert x <10.0
            assert not y <0.0
            
        def test_lesseq():
            assert x <= z
            assert not x<=y
            assert z <= 4.0
            assert y <= 10.0
            
        def test_greater():
            assert x > y
            assert not y >z
            assert x+2.0 > z
            
        def test_greatereq():
            assert x >=z 
            assert x>=y
            assert y+10.0 >= x
            assert not y >=z
        '''
        

        test_eq()
        test_ne()
        '''
        test_less()
        test_lesseq()
        test_greater()
        test_greatereq()
        '''

    def test_composition():
        x = Var(np.pi)

        def test_1_trig():
            z1 = Var.sin(Var.cos(x))
            assert np.round(z1.get_value(),10) == -0.8414709848
            assert np.round(z1.get_der(),10) == [0]

        def test_2():
            z2 = Var.sin(x**2)
            assert np.round(z2.get_value(), 9) == -0.430301217
            assert np.round(z2.get_der(),10) == -5.6717394031
        test_1_trig()
        test_2()

    test_overloading()
    test_comparison()
    test_composition()
    
def test_vector_scalar():
    def test_overloading():
        x = Var(4.0)
        y = Var(3.0)
        z = Var(5.0)
        def test_add():
            z1 = x+y
            assert z1.get_value() == 7.0
            assert z1.get_der() == [1.0, 1.0]
            assert z1._get_derivative_of(x) == 1.0
            assert z1._get_derivative_of(y) == 1.0

        def test_radd():
            z1 = 2+y+10+x
            assert z1.get_value() == 19.0
            assert z1.get_der() == [1.0, 1.0]
            assert z1._get_derivative_of(x) == 1.0
            assert z1._get_derivative_of(y) == 1.0

        def test_subtract():
            z1 = z-x-y-1
            assert z1.get_value() == -3.0
            assert z1.get_der([x, y, z]) == [-1.0, -1.0, 1.0]
            assert z1._get_derivative_of(x) == -1.0
            assert z1._get_derivative_of(y) == -1.0
            assert z1._get_derivative_of(z) == 1.0

            # z not modified
            assert z.get_value() == Var(5.0).get_value()

        def test_rsubtract():
            z1 = 10.0-x-y
            assert z1.get_value() == 3.0
            assert z1.get_der() == [-1.0, -1.0]
            assert z1._get_derivative_of(x) == -1.0
            assert z1._get_derivative_of(y) == -1.0

        def test_mul():
            z1 = x*y*z*10
            assert z1.get_value() == 600.0
            assert z1.get_der() == [150.0, 200.0, 120.0]
            assert z1._get_derivative_of(x) == 150.0
            assert z1._get_derivative_of(y) == 200.0
            assert z1._get_derivative_of(z) == 120.0

        def test_rmul():
            z1 = 2*x*5*y*z
            assert z1.get_value() == 600.0
            assert z1.get_der() == [150.0, 200.0, 120.0]
            assert z1._get_derivative_of(x) == 150.0
            assert z1._get_derivative_of(y) == 200.0
            assert z1._get_derivative_of(z) == 120.0

        def test_div():
            z1 = (z/x)/10
            assert z1.get_value() == 5.0/40.0
            assert z1.get_der([x,z]) == [ -5.0/160.0, 1.0/40.0]
            assert z1._get_derivative_of(z) == 1.0/40.0
            assert z1._get_derivative_of(x) == -5.0/160.0

        def test_rdiv():
            z1 = 6.0/(y/x)
            assert z1.get_value() == 8.0
            assert z1.get_der() == [-8.0/3.0, 2.0]  #dz1/dy first, dz1/dx second
            assert z1._get_derivative_of(y) == -8.0/3.0
            assert z1._get_derivative_of(x) == 2.0

        def test_pow():
            z1 = x**y
            assert z1.get_value() == 64.0
            np.testing.assert_array_equal(np.round(z1.get_der([x,y]),10), np.round([48.0, 64*np.log(4)], 10))
            assert np.round(z1._get_derivative_of(x), 10) == np.round(48.0, 10)
            assert np.round(z1._get_derivative_of(y), 10) == np.round(64*np.log(4), 10)

        def test_rpow():
            z1 = (2**x)**y
            assert z1.get_value() == 4096.0
            np.testing.assert_array_equal(np.round(z1.get_der(), 10), np.round([12288*np.log(2), 16384*np.log(2)], 10))
            assert np.round(z1._get_derivative_of(x), 10) == np.round(12288*np.log(2), 10)
            assert np.round(z1._get_derivative_of(y), 10) == np.round(16384*np.log(2), 10)

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



    
    def test_comparison():
        x = Var(4.0)
        y = Var(1.5)
        z1 = x + y
        z2 = abs(x+y)
            
        def test_eq():
            assert z1 == z2
            assert z1.__eq__(z2)
            assert not x == y
                
            m1 = x*4.0/2.0
            m2 = x*2.0
            assert m1 == m2
            
        def test_ne():
            z3 = -z1
            assert not z3 == z2
            
            x2 = -x
            x2.get_value() == -4.0
            
            
        '''    
        def test_less():
            assert x > y
            assert not z < y
            assert x <10.0
            assert not y <0.0
            
        def test_lesseq():
            assert x <= z
            assert not x<=y
            assert z <= 4.0
            assert y <= 10.0
            
        def test_greater():
            assert x > y
            assert not y >z
            assert x+2.0 > z
            
        def test_greatereq():
            assert x >=z 
            assert x>=y
            assert y+10.0 >= x
            assert not y >=z
        '''    
        test_eq()
        test_ne()
        '''
        test_less()
        test_lesseq()
        test_greater()
        test_greatereq()
        '''

    def test_composition():
        x = Var(np.pi)
        y = Var(0)
        z = Var(np.pi/2)

        def test_1_trig():
            z1 = Var.sin(Var.cos(x))+ Var.sin(y)
            assert np.round(z1.get_value(),10) == -0.8414709848
            np.testing.assert_array_equal(np.round(z1.get_der([x,y]), 10), np.round([0, 1.0], 10))
            assert np.round(z1._get_derivative_of(x), 10) == np.round(0, 10)
            assert np.round(z1._get_derivative_of(y), 10) == np.round(1.0, 10)
            

        def test_2():
            z2 = Var.sin(x**2) + Var.sin(z)
            assert np.round(z2.get_value(), 9) == -0.430301217+1.0
            np.testing.assert_array_equal(np.round(z2.get_der([x,z]),10), np.round([-5.6717394031, 0], 10))
            
        test_1_trig()
        test_2()

    test_overloading()
    test_comparison()
    test_composition()

def test_scalar_vector():
    def test_overloading():
        x = Var(4.0)
        y = MultiFunc([2*x, x**2])
        z = MultiFunc([x, 2*x])
        def test_add():
            z1 = y + 1
            assert z1.get_value() == [9.0, 17.0]
            assert z1.get_der([x]) == [[2.0], [8.0]]
            
            z2 = z + MultiFunc([x, x])
            assert z2.get_value() == [8.0, 12.0]
            assert z2.get_der([x]) == [[2.0], [3.0]]

        def test_radd():
            z1 = 2 + y
            assert z1.get_value() == [10.0, 18.0]
            assert z1.get_der([x]) == [[2.0], [8.0]]

        def test_subtract():
            z1 = y - z - 1  # = [x, x^2 - 2x]
            assert z1.get_value() == [3.0, 7.0]
            assert z1.get_der([x]) == [[1.0], [6.0]]

            # z not modified
            assert z.get_value() == MultiFunc([x, 2*x]).get_value()

        def test_rsubtract():
            z1 = 10.0-y-z
            assert z1.get_value() == [-2.0, -14.0]
            assert z1.get_der([x]) == [[-3.0], [-10.0]]

        def test_mul():
            z1 = y*z*2
            assert z1.get_value() == [64.0, 256.0]
            assert z1.get_der([x]) == [[32.0], [192.0]]

        def test_rmul():
            z1 = 2*y*z
            assert z1.get_value() == [64.0, 256.0]
            assert z1.get_der([x]) == [[32.0], [192.0]]

        def test_div():
            z1 = (y/z)/10
            assert z1.get_value() == [0.2, 0.2]
            assert z1.get_der([x]) == [[0.0], [0.05]]

        def test_rdiv():
            z1 = 10/(y/x)
            assert z1.get_value() == [5.0, 2.5]
            assert z1.get_der([x]) == [[0], [-0.625]]

        def test_pow():
            z1 = y**2
            assert z1.get_value() == [64.0, 256.0]
            assert z1.get_der([x]) == [[32.0], [256.0]]  #=【8x, 4x^3]

        def test_rpow():
            z1 = 2**y
            assert z1.get_value() == [256.0, 65536.0]
            np.testing.assert_array_equal(np.round(z1.get_der([x]), 10), np.round([[512.0*np.log(2)], [524288.0*np.log(2)]], 10))

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



    
    def test_comparison():
        x = Var(4.0)
        y = MultiFunc([x**2, x+x])
        z = MultiFunc([x*x, 2*x])
        
        def test_eq():
            assert y == z
            assert y.__eq__(z)
                
            y1 = y*4.0/2.0
            z1 = z*2.0
            assert y1 == z1
        
        def test_ne():
            x1 = -x
            x1.get_value() == -4.0
            y1 = -y
            y1.get_value() == [-16.0, -8.0]
        
        '''    
        def test_less():
            assert x > y
            assert not z < y
            assert x <10.0
            assert not y <0.0
            
        def test_lesseq():
            assert x <= z
            assert not x<=y
            assert z <= 4.0
            assert y <= 10.0
            
        def test_greater():
            assert x > y
            assert not y >z
            assert x+2.0 > z
            
        def test_greatereq():
            assert x >=z 
            assert x>=y
            assert y+10.0 >= x
            assert not y >=z
        '''
        
        test_eq()
        test_ne()
        '''
        test_less()
        test_lesseq()
        test_greater()
        test_greatereq()
        '''

    def test_composition():
        x = Var(np.pi)
        y = MultiFunc([x, x])
        def test_1_trig():
            f = y.apply(Var.cos)
            z1 = f.apply(Var.sin)
            np.testing.assert_array_equal(np.round(z1.get_value(),10), np.round([-0.8414709848, -0.8414709848], 10))
            np.testing.assert_array_equal(np.round(z1.get_der([x]),10), np.round([[0], [0]], 10))
            

        '''
        def test_2():
            z2 = Var.sin(x**2) + Var.sin(z)
            assert np.round(z2.get_value(), 9) == -0.430301217+1.0
            assert np.round(z2.get_der(),10) == [-5.6717394031, 0]
        '''   
        test_1_trig()


    test_overloading()
    test_comparison()
    test_composition()

def test_vector_vector():
    def test_overloading():
        x = Var(3.0)
        y = Var(2.0)
        z = MultiFunc([x, y])
        p = MultiFunc([y*2, x+2])
        def test_add():
            z1 = z + 1
            assert z1.get_value() == [4.0, 3.0]
            assert z1.get_der([x, y]) == [[1.0, 0], [0, 1.0]]
            assert z1.get_der([x]) == [[1.0], [0]]
            assert z1.get_der([y]) == [[0], [1.0]]
            #assert z1._get_derivative_of(x) = 1.0
            #assert z1._get_derivative_of(x2) = 8.0
            
            z2 = MultiFunc([y, x]) + MultiFunc([x, x])
            assert z2.get_value() == [5.0, 6.0]
            assert z2.get_der([x,y]) == [[1.0, 1.0], [2.0, 0]]
            assert z2.get_der([x]) == [[1.0], [2.0]]
            assert z2.get_der([y]) == [[1.0], [0]]

        def test_radd():
            z1 = 1 + z
            assert z1.get_value() == [4.0, 3.0]
            assert z1.get_der([x, y]) == [[1.0, 0], [0, 1.0]]
            assert z1.get_der([x]) == [[1.0], [0]]
            assert z1.get_der([y]) == [[0], [1.0]]

        def test_subtract():
            z1 = z - 1  # = [x, x^2 - 2x]
            assert z1.get_value() == [2.0, 1.0]
            assert z1.get_der([x, y]) == [[1.0, 0], [0, 1.0]]
            assert z1.get_der([x]) == [[1.0], [0]]
            assert z1.get_der([y]) == [[0], [1.0]]
            
            # z not modified
            assert z.get_value() == MultiFunc([Var(3.0), Var(2.0)]).get_value()
            
            z2 = z - MultiFunc([y, x])
            assert z2.get_value() == [1.0, -1.0]
            assert z2.get_der([x,y]) == [[1.0, -1.0], [-1.0, 1.0]]
            assert z2.get_der([x]) == [[1.0], [-1.0]]
            assert z2.get_der([y]) == [[-1.0], [1.0]]

        def test_rsubtract():
            z1 = 10.0-z
            assert z1.get_value() == [7.0, 8.0]
            assert z1.get_der([x, y]) == [[-1.0, 0], [0, -1.0]]
            assert z1.get_der([x]) == [[-1.0], [0]]
            assert z1.get_der([y]) == [[0], [-1.0]]

        def test_mul():
            z1 = z*2
            assert z1.get_value() == [6.0, 4.0]
            assert z1.get_der([x,y]) == [[2.0, 0], [0, 2.0]]
            assert z1.get_der([x]) == [[2.0], [0]]
            assert z1.get_der([y]) == [[0], [2.0]]
            
            z2 = z*p  # = [x*y^2, y(x+2)]
            assert z2.get_value() == [12.0, 10.0]
            assert z2.get_der([x,y]) == [[4.0, 6.0], [2.0, 5.0]]
            assert z2.get_der([x]) == [[4.0], [2.0]]
            assert z2.get_der([y]) == [[6.0], [5.0]]

        def test_rmul():
            z1 = 2*z
            assert z1.get_value() == [6.0, 4.0]
            assert z1.get_der([x,y]) == [[2.0, 0], [0, 2.0]]
            assert z1.get_der([x]) == [[2.0], [0.0]]
            assert z1.get_der([y]) == [[0], [2.0]]

        def test_div():
            z1 = z/10
            assert z1.get_value() == [0.3, 0.2]
            assert z1.get_der([x,y]) == [[0.1, 0], [0, 0.1]]
            assert z1.get_der([x]) == [[0.1], [0]]
            assert z1.get_der([y]) == [[0], [0.1]]
            
            z2 = p/z
            np.testing.assert_array_equal(np.round(z2.get_value(), 10), np.round([4.0/3, 2.5], 10))
            np.testing.assert_array_equal(np.round(z2.get_der([x,y]), 10), np.round([[-4.0/9, 2.0/3], [0.5, -5.0/4]], 10))
            np.testing.assert_array_equal(np.round(z2.get_der([x]),10), np.round([[-4.0/9], [0.5]], 10))
            np.testing.assert_array_equal(np.round(z2.get_der([y]), 10), np.round([[2.0/3], [-5.0/4]], 10))

        def test_rdiv():
            z1 = 10/z
            np.testing.assert_array_equal(np.round(z1.get_value(), 10), np.round([10.0/3, 5.0], 10))
            np.testing.assert_array_equal(np.round(z1.get_der([x,y]), 10), np.round([[-10.0/9, 0], [0, -2.5]], 10))
            np.testing.assert_array_equal(np.round(z1.get_der([x]),10), np.round([[-10.0/9], [0]], 10))
            np.testing.assert_array_equal(np.round(z1.get_der([y]), 10), np.round([[0], [-2.5]], 10))

        def test_pow():
            z1 = z**2
            assert z1.get_value() == [9.0, 4.0]
            assert z1.get_der([x,y]) == [[6.0, 0], [0, 4.0]] 
            assert z1.get_der([x]) == [[6.0], [0]]
            assert z1.get_der([y]) == [[0], [4.0]]
            
            z2 = z**p
            assert z2.get_value() == [81.0, 32.0]
            np.testing.assert_array_equal(np.round(z2.get_der([x,y]), 10), np.round([[108.0, 162.0*np.log(3)], [32.0*np.log(2), 80.0]], 10))
            np.testing.assert_array_equal(np.round(z2.get_der([x]),10), np.round([[108.0], [32.0*np.log(2)]], 10))
            np.testing.assert_array_equal(np.round(z2.get_der([y]), 10), np.round([[162.0*np.log(3)], [80.0]], 10))

        def test_rpow():
            z1 = 2**z
            assert z1.get_value() == [8.0, 4.0]
            np.testing.assert_array_equal(np.round(z1.get_der([x,y]), 10), np.round([[8.0*np.log(2), 0], [0, 4.0*np.log(2)]], 10))
            np.testing.assert_array_equal(np.round(z1.get_der([x]),10), np.round([[8.0*np.log(2)], [0]], 10))
            np.testing.assert_array_equal(np.round(z1.get_der([y]), 10), np.round([[0], [4.0*np.log(2)]], 10))
            

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



    
    def test_comparison():
        x = Var(4.0)
        y = Var(1.5)
        temp = Var(0.5)
        z = MultiFunc([x+y+y, x-y-y])
        p = MultiFunc([x+2*y, x-2*y])
        q = MultiFunc([x*2-temp, x-y-y])
            
        def test_eq():
            assert p == z
            assert p.__eq__(z)
            assert not z == q
                
            z1 = z*4.0/2.0
            p1 = p*2.0
            assert z1 == p1
        
        def test_ne():
            z1 = -z
            p1 = -p
            assert z1 == p1
            z1.get_value() == [7.0, 1.0]
            z1.get_der([x,y]) == [[1.0, 2.0], [1.0, -2.0]]
        
        '''    
        def test_less():
            assert x > y
            assert not z < y
            assert x <10.0
            assert not y <0.0
            
        def test_lesseq():
            assert x <= z
            assert not x<=y
            assert z <= 4.0
            assert y <= 10.0
            
        def test_greater():
            assert x > y
            assert not y >z
            assert x+2.0 > z
            
        def test_greatereq():
            assert x >=z 
            assert x>=y
            assert y+10.0 >= x
            assert not y >=z
        '''
        test_eq()
        test_ne()

    def test_composition():
        

        def test_1_trig():
            x = Var(np.pi)
            y = Var(np.pi/2)
            z = MultiFunc([x, y])
            f = z.apply(Var.cos)
            z1 = f.apply(Var.sin)
            np.testing.assert_array_equal(np.round(z1.get_value(),10), np.round([-0.8414709848, 0], 10))
            np.testing.assert_array_equal(np.round(z1.get_der([x,y]),10), np.round([[0, 0], [0, -1.0]], 10))  # = -sin(x)cos(cos(x)) - sin(y)cos(cos(y))
            np.testing.assert_array_equal(np.round(z1.get_der([x]), 10), np.round([[0], [0]], 10))
            assert z1.get_der([y]) == [[0], [-1.0]]
        
        def test_2():
            x = Var(1.0)
            y = Var(2.0)
            z = MultiFunc([x, y])  
            z1 = z**2+2
            assert z1.get_value() == [3.0, 6.0]    
            assert z1.get_der([x,y]) == [[2.0,0], [0, 4.0]]
            assert z1.get_der([x]) == [[2.0], [0]]
            assert z1.get_der([y]) == [[0], [4.0]]
              
        test_1_trig()
        test_2()


    test_overloading()
    test_comparison()
    test_composition()
        

test_constructor()
test_scalar_scalar()
test_vector_scalar()
test_scalar_vector()
test_vector_vector()

# def test_vector_input():
    #     #x = Var(np.array([4.0, [1.0,0]]))
    #     #y = Var(np.array([2.0, [0, 1.0]]))
    #     def suite_add():
    #         x = Var(np.array([4.0]))
    #         y = Var(np.array([1.5]))
    #         z1 = x+y
    #         assert z1.get_value() == [5.5]
    #         assert z1.get_der() = [1.0, 1.0]
    #
    #         z2 = x+2+y+10
    #         assert z2.get_value() == [20.0]
    #         assert z2.get_der() = [1.0, 1.0]
    #
    #         # x not modified
    #         assert x == Var(np.array([4.0]))
    #
    #     def suite_radd():
    #         x = Var(np.array([4.0]))
    #         z1 = 2+x+10+x
    #         assert z1.get_value() == [20.0]
    #         assert z1.get_der() = [2.0]
    #
    #         z2 = 1.0+(3.0+2.0)+x+x
    #         assert z2 = [14.0]
    #         assert z1.get_der() = [2.0]
    #
    #     def suite_subtract():
    #         x = Var(np.array([4.0]))
    #         y = Var(np.array([1.5]))
    #         z1 = x-y-1.0
    #         assert z1.get_value() == [1.5]
    #         assert z1.get_der() = [1.0, -1.0]
    #
    #         z2 = x -1.0 -2.0 -x.0
    #         assert z2.get_value() == [-3.0]
    #         assert z2.get_der() = [0]
    #
    #         # x not modified
    #         assert x == Var(np.array([4.0]))
    #
    #     def suite_rsubtract():
    #         x = Var(np.array([4.0]))
    #         z1 = 10.0-x
    #         assert z1.get_value() == [6.0]
    #         assert z1.get_der() = [-1.0]
    #
    #         z2 = 20.0-3.0-x-x
    #         assert z2.get_value() == [9.0]
    #         assert z2.get_der() = [-2.0]
    #
    #     def suite_mul():
    #         x = Var(np.array([4.0]))
    #         y = Var(np.array([2.0]))
    #         z1 = x*y
    #         assert z1.get_value() == [8.0]
    #         assert z1.get_der() == [2.0, 4.0]
    #
    #         z2 = x*2*3
    #         assert z2.get_value() == [24.0]
    #         assert z2.get_der() == 6.0
    #
    #         z3 = x*x+x*2
    #         assert z3.get_value() == [24.0]
    #         assert z3.get_der() == [10.0]
    #
    #         # x not modified
    #         assert x == Var(np.array([4.0]))
    #
    #     def suite_rmul():
    #         x = Var(np.array([4.0]))
    #         z1 = 3*x
    #         assert z1.get_value()  == [12.0]
    #         assert z1.get_der() = [3.0]
    #
    #         z1 = 3*10*x*x
    #         assert z2.get_value() == [480.0]
    #         assert z2.get_der() = [240.0]
    #
    #     def suite_div():
    #         z1 = x/y
    #         assert z1.get_value() == [2.0]
    #         assert z1.get_der() == [0.5, -1.0]
    #
    #         z2 = x/4
    #         assert z2.get_value() == [1.0]
    #         assert z2.get_der() = [0.25]
    #
    #         z3 = (x/0.5)/0.1
    #         assert z3.get_value() == [80.0]
    #         assert z3.get_der() = [20.0]
    #
    #     def suite_rdiv():
    #         z1 = 8.0/x
    #         assert z1.get_value() == [2.0]
    #         assert z1.get_der() = [-0.5]
    #
    #         z2 = (24.0/x)/1.5
    #         assert z2.get_value() == [4.0]
    #         assert z2.get_der() == [-1.0]
    #
    #     def suite_pow():
    #         z1 = x**2
    #         assert z1.get_value() == [16.0]
    #         assert z1.get_der() = [8.0]
    #         assert z1 = x*x
    #
    #         z2 = x**(0.5)
    #         assert z2.get_value() == [2.0]
    #         assert z2.get_der() == [0.25]
    #         assert z2 = np.sqrt(x)
    #
    #     def suite_rpow():
    #         # (a^x)' = lna*(a^x)
    #         a = 2
    #         z1 = a**x
    #         assert z1.get_value() == [16.0]
    #         assert z1.get_der() == [np.log(a)*16.0]
    #
    #         z2 = a**(x+y)
    #         assert z2.get_value() == [64.0]
    #         assert z2.get_der() == [np.log(a)*64.0, np.log(a)*64.0]

# def overwritten_comparison():
#     x = Var(4.0)
#     y = Var(1.5)
#     z = Var(4.0)
#
#     def suite_eq():
#         assert x.get_value() == z.get_value()
#         assert x.__eq__(z)
#         assert not x == y
#
#         z1 = x*4.0/2.0
#         z2 = z*2.0
#         assert z1 == z2
#
#     def suite_less():
#         assert x > y
#         assert not z < y
#         assert x <10.0
#         assert not y <0.0
#
#     def suite_lesseq():
#         assert x <= z
#         assert not x<=y
#         assert z <= 4.0
#         assert y <= 10.0
#
#     def suite_greater():
#         assert x > y
#         assert not y >z
#         assert x+2.0 > z
#
#     def suite_greatereq():
#         assert x >=z
#         assert x>=y
#         assert y+10.0 >= x
#         assert not y >=z
#
#     suite_eq()
#     suite_greater()
#     suite_greatereq()
#     suite_less()
#     suite_lesseq()

