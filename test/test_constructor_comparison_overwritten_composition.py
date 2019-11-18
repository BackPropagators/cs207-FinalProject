import AutoDiff.ForwardAd as Var
import numpy as np
import math

#input is an array, output is a list
def test_constructor():
    def test_np_input():
        x = Var(np.array([1.0]))
        assert x.val == [1.0]  
        assert type(x.val) == np.ndarray
        assert x.jacobian = [1.0]
        
    def test_with_given_jac():   
        jac = np.array[3.2]
        x = Var(np.array[2.0], jac)
        assert x.val == [2.0]
        assert type(x.val) == np.ndarray
        assert x.jacobian == [jac[0]] 

        
def overwritten_elementary():
    def test_scalar_input():
        x = Var(np.array([4.0]))
        y = Var(np.array([2.0]))
        def suite_add():
            x = Var(np.array([4.0]))
            y = Var(np.array([1.5]))
            z1 = x+y
            assert z1.val == [5.5]
            assert z1.jacobian = [1.0, 1.0]
            
            z2 = x+2+x+10
            assert z2.val == [20.0]
            assert z2.jacobian = [2.0]
            
            # x not modified
            assert x == Var(np.array([4.0])) 
            
        def suite_radd():
            x = Var(np.array([4.0]))
            z1 = 2+x+10+x
            assert z1.val == [20.0]
            assert z1.jacobian = [2.0]
            
            z2 = 1.0+(3.0+2.0)+x+x
            assert z2 = [14.0]
            assert z1.jacobian = [2.0]
            
        def suite_subtract():
            x = Var(np.array([4.0]))
            y = Var(np.array([1.5]))
            z1 = x-y-1.0
            assert z1.val == [1.5]
            assert z1.jacobian = [1.0, -1.0]
            
            z2 = x -1.0 -2.0 -x.0
            assert z2.val == [-3.0]
            assert z2.jacobian = [0]
            
            # x not modified
            assert x == Var(np.array([4.0]))
            
        def suite_rsubtract():
            x = Var(np.array([4.0]))
            z1 = 10.0-x
            assert z1.val == [6.0]
            assert z1.jacobian = [-1.0]
            
            z2 = 20.0-3.0-x-x
            assert z2.val == [9.0]
            assert z2.jacobian = [-2.0]
            
        def suite_mul():
            x = Var(np.array([4.0]))
            y = Var(np.array([2.0]))
            z1 = x*y
            assert z1.val == [8.0]
            assert z1.jacobian == [2.0, 4.0]
            
            z2 = x*2*3
            assert z2.val == [24.0]
            assert z2.jacobian == 6.0
            
            z3 = x*x+x*2
            assert z3.val == [24.0]
            assert z3.jacobian == [10.0]
            
            # x not modified
            assert x == Var(np.array([4.0]))
            
        def suite_rmul():
            x = Var(np.array([4.0]))
            z1 = 3*x
            assert z1.val  == [12.0]
            assert z1.jacobian = [3.0]
            
            z1 = 3*10*x*x
            assert z2.val == [480.0]
            assert z2.jacobian = [240.0]
            
        def suite_div():
            z1 = x/y
            assert z1.val == [2.0]
            assert z1.jacobian == [0.5, -1.0]
            
            z2 = x/4
            assert z2.val == [1.0]
            assert z2.jacobian = [0.25]
            
            z3 = (x/0.5)/0.1
            assert z3.val == [80.0]
            assert z3.jacobian = [20.0]
            
        def suite_rdiv():
            z1 = 8.0/x
            assert z1.val == [2.0]
            assert z1.jacobian = [-0.5]
            
            z2 = (24.0/x)/1.5
            assert z2.val == [4.0]
            assert z2.jacobian == [-1.0]
            
        def suite_pow():
            z1 = x**2
            assert z1.val == [16.0]
            assert z1.jacobian = [8.0]
            assert z1 = x*x
            
            z2 = x**(0.5)
            assert z2.val == [2.0]
            assert z2.jacobian == [0.25]
            assert z2 = np.sqrt(x)
            
        def suite_rpow():
            # (a^x)' = lna*(a^x)
            a = 2
            z1 = a**x
            assert z1.val == [16.0]
            assert z1.jacobian == [np.log(a)*16.0]
            
            z2 = a**(x+y)
            assert z2.val == [64.0]
            assert z2.jacobian == [np.log(a)*64.0, np.log(a)*64.0]
    
    '''       
    def test_vector_input():
        #x = Var(np.array([4.0, [1.0,0]]))
        #y = Var(np.array([2.0, [0, 1.0]]))
        def suite_add():
            x = Var(np.array([4.0]))
            y = Var(np.array([1.5]))
            z1 = x+y
            assert z1.val == [5.5]
            assert z1.jacobian = [1.0, 1.0]
            
            z2 = x+2+y+10
            assert z2.val == [20.0]
            assert z2.jacobian = [1.0, 1.0]
            
            # x not modified
            assert x == Var(np.array([4.0])) 
            
        def suite_radd():
            x = Var(np.array([4.0]))
            z1 = 2+x+10+x
            assert z1.val == [20.0]
            assert z1.jacobian = [2.0]
            
            z2 = 1.0+(3.0+2.0)+x+x
            assert z2 = [14.0]
            assert z1.jacobian = [2.0]
            
        def suite_subtract():
            x = Var(np.array([4.0]))
            y = Var(np.array([1.5]))
            z1 = x-y-1.0
            assert z1.val == [1.5]
            assert z1.jacobian = [1.0, -1.0]
            
            z2 = x -1.0 -2.0 -x.0
            assert z2.val == [-3.0]
            assert z2.jacobian = [0]
            
            # x not modified
            assert x == Var(np.array([4.0]))
            
        def suite_rsubtract():
            x = Var(np.array([4.0]))
            z1 = 10.0-x
            assert z1.val == [6.0]
            assert z1.jacobian = [-1.0]
            
            z2 = 20.0-3.0-x-x
            assert z2.val == [9.0]
            assert z2.jacobian = [-2.0]
            
        def suite_mul():
            x = Var(np.array([4.0]))
            y = Var(np.array([2.0]))
            z1 = x*y
            assert z1.val == [8.0]
            assert z1.jacobian == [2.0, 4.0]
            
            z2 = x*2*3
            assert z2.val == [24.0]
            assert z2.jacobian == 6.0
            
            z3 = x*x+x*2
            assert z3.val == [24.0]
            assert z3.jacobian == [10.0]
            
            # x not modified
            assert x == Var(np.array([4.0]))
            
        def suite_rmul():
            x = Var(np.array([4.0]))
            z1 = 3*x
            assert z1.val  == [12.0]
            assert z1.jacobian = [3.0]
            
            z1 = 3*10*x*x
            assert z2.val == [480.0]
            assert z2.jacobian = [240.0]
            
        def suite_div():
            z1 = x/y
            assert z1.val == [2.0]
            assert z1.jacobian == [0.5, -1.0]
            
            z2 = x/4
            assert z2.val == [1.0]
            assert z2.jacobian = [0.25]
            
            z3 = (x/0.5)/0.1
            assert z3.val == [80.0]
            assert z3.jacobian = [20.0]
            
        def suite_rdiv():
            z1 = 8.0/x
            assert z1.val == [2.0]
            assert z1.jacobian = [-0.5]
            
            z2 = (24.0/x)/1.5
            assert z2.val == [4.0]
            assert z2.jacobian == [-1.0]
            
        def suite_pow():
            z1 = x**2
            assert z1.val == [16.0]
            assert z1.jacobian = [8.0]
            assert z1 = x*x
            
            z2 = x**(0.5)
            assert z2.val == [2.0]
            assert z2.jacobian == [0.25]
            assert z2 = np.sqrt(x)
            
        def suite_rpow():
            # (a^x)' = lna*(a^x)
            a = 2
            z1 = a**x
            assert z1.val == [16.0]
            assert z1.jacobian == [np.log(a)*16.0]
            
            z2 = a**(x+y)
            assert z2.val == [64.0]
            assert z2.jacobian == [np.log(a)*64.0, np.log(a)*64.0]
     '''   
def overwritten_comparison():
    x = Var(np.array([4.0]))
    y = Var(np.array([1.5]))
    z = Var(np.array([4.0]))
        
    def suite_eq():
        assert x == z
        assert x.__eq__(z)
        assert not x == y
            
        z1 = x*4.0/2.0
        z2 = z*2.0
        assert z1 == z2
        
    def suite_less():
        assert x > y
        assert not z < y
        assert x <10.0
        assert not y <0.0
        
    def suite_lesseq():
        assert x <= z
        assert not x<=y
        assert z <= 4.0
        assert y <= 10.0
        
    def suite_greater():
        assert x > y
        assert not y >z
        assert x+2.0 > z
        
    def suite_greatereq():
        assert x >=z 
        assert x>=y
        assert y+10.0 >= x
        assert not y >=z


def composition():
    x = Var(np.array([np.pi]))
    y = Var(np.array([0]))
    z = Var(np.array([np.pi]))
    
    def trig():
        z1 = np.sin(np.cos(x)+np.sin(y))
        assert np.round(z1.val,11) == [-0.8414709848]
        assert np.round(z1.jacobian,11) == [0, 0.54030230586]
        

#def test_m_to_n():
    
        
            
        
        