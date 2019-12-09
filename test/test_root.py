from AutoDiff.ForwardAD import Var, MultiFunc
import numpy as np
from AutoDiff.root_finding import Newton

def test_root():
    f1 = lambda x: x**2-1
    guess1 = 2
    assert Newton(f1, guess1) == 1.0
    
    f2 = lambda x: x**3 -1
    guess2 = 2
    assert Newton(f2, guess2) == 1.0
    
    f3 = lambda x: x+10
    guess3 = -8
    assert Newton(f3, guess3) == -10.0
    
test_root()