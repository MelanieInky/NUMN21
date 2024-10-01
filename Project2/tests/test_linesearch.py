""" Tests for the different line searches"""
import unittest
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from Project2.finite_diff import finite_difference
from Project2.optimization import OptimizationProblem
from Project2.optimizer import NewtonOptimizer
import numpy as np

def f(x):
    return x**2

def gradf(x):
    return 2*x

def g(x):
    return x[0]**2 + (x[1] + 1)**2

def gradg(x):
    return np.array([2*x[0],2*(x[1]+1)])

def h(x):
    return x[0] * x[1]

def gradh(x):
    return np.array([x[1],x[0]])

class TestANN(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.problem = OptimizationProblem(f,gradf)
        self.opt = NewtonOptimizer(self.problem)
        self.problem2d = OptimizationProblem(g,gradg)
        self.opt2d = NewtonOptimizer(self.problem2d)
        self.problem2d_2 = OptimizationProblem(h,gradh)
        self.opt2d_2 = NewtonOptimizer(self.problem2d_2)
        
    def test_exact_line_search(self):
        print("\n---- Running tests for exact line search, part 1 -----")
        x = 2
        s = -1
        alpha = self.opt.exact_line_search(x,s)
        print(f'Expecting alpha = {x}, received alpha = {alpha}')
        self.assertAlmostEqual(alpha,2)
    def test_exact_line_search2(self):
        print("\n---- Running tests for exact line search, part 2-----")
        x = 4
        s = -1
        alpha = self.opt.exact_line_search(x,s)
        print(f'Expecting alpha = {x}, received alpha = {alpha}')
        self.assertAlmostEqual(alpha,4)
        
    
    @unittest.expectedFailure    
    def test_exact_line_search3(self):
        print("\n---- Running tests for exact line search, part 3-----")
        x = -2
        s = -1
        alpha = self.opt.exact_line_search(x,s)
        print(f'Expecting failure, as alpha cannot be negative')
        self.assertAlmostEqual(alpha,-2)
    
    def test_exact_line_search4(self):
        print("\n---- Running tests for exact line search, part 4-----")
        x = np.array([1,1])
        s = np.array([-1,-1])
        alpha = self.opt2d.exact_line_search(x,s)
        expected_alpha = 1/2*(x[0] + x[1] + 1)
        print(f'Expecting alpha = {expected_alpha}, received alpha = {alpha}')
        self.assertAlmostEqual(alpha,expected_alpha)
        
    def test_exact_line_search5(self):
        print("\n---- Running tests for exact line search, part 5-----")
        x = np.array([-3,1])
        s = np.array([1,-1])
        alpha = self.opt2d_2.exact_line_search(x,s)
        expected_alpha = -1/(2*s[0]*s[1])*(s[0] * x[1] + s[1]*x[0])
        print(f'Expecting alpha = {expected_alpha}, received alpha = {alpha}')
        self.assertAlmostEqual(alpha,expected_alpha)

        

unittest.main(argv=['first-arg-is-ignored'], exit=False)