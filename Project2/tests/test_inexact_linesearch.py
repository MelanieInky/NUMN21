""" Tests for the different line searches"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import unittest

from Project2.finite_diff import finite_difference
from Project2.optimization import OptimizationProblem
from Project2.optimizer import NewtonOptimizer
import numpy as np
from Project2.line_search_visualizer import plot_phi
import matplotlib.pyplot as plt

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


def wolfe_powell(f,fgrad,x,s,alpha, rho, sigma):
    """Returns true if the wolfe powell condition are fullfilled

    Args:
        f (_type_): The function
        fgrad (_type_): Gradient of the function
        x (_type_): Value of x
        s (_type_): search direction
        alpha (_type_): search direction multiplier
        rho (_type_): rho in Wolfe-Powell conditions
        sigma (_type_): sigmain Wolfe-Powell conditions
    """
    def dphi(alpha):
        return np.dot(s,fgrad(x + alpha * s))
    def phi(alpha):
        return f(x + alpha * s)
    cond1 = phi(alpha) <= phi(0) + alpha*rho*dphi(0)
    cond2 = dphi(alpha) >= sigma * dphi(0)
    return cond1 and cond2

class TestANN(unittest.TestCase):

    
    
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.problem = OptimizationProblem(f,gradf)
        self.opt = NewtonOptimizer(self.problem)
        self.problem2d = OptimizationProblem(g,gradg)
        self.opt2d = NewtonOptimizer(self.problem2d)
        self.problem2d_2 = OptimizationProblem(h,gradh)
        self.opt2d_2 = NewtonOptimizer(self.problem2d_2)
        
    @unittest.expectedFailure
    def test_initialization(self):
        #If not set up, it should fail
        opt = self.opt
        rho = 0.01
        sigma = 0.3
        x = np.array([2])
        s = np.array([-1])
        opt.inexact_line_search(x,s)


    def test_bad_lower_bound(self):
        print("Testing if setting a bad lower bound returns 0")
        opt = self.opt
        rho = 0.01
        sigma = 0.3
        x = np.array([0.5])
        s = np.array([-1])
        opt.setup_inexact_line_search(1,rho = rho, sigma = sigma,alpha_init=1)
        alpha = opt.inexact_line_search(x,s)
        assert(alpha == 0)
    
    def test_inexact_line_search(self):
        print("Testing that inexact Powell-Wolfe condition are fulfilled, part1")
        opt = self.opt
        rho = 0.01
        sigma = 0.3
        opt.setup_inexact_line_search(0,rho = rho, sigma = sigma,alpha_init=1)
        x = np.array([2])
        s = np.array([-1])
        alpha = opt.inexact_line_search(x,s)
        cond = wolfe_powell(f,gradf,x,s,alpha,rho,sigma)
        assert(cond == True)
        

    def test_inexact_line_search2(self):
        print("Testing that inexact Powell-Wolfe condition are fullfilled, part2")
        #Fails because alpha does not grow fast enough
        #It is, in my opinion, an issue
        opt = self.opt
        rho = 0.01
        sigma = 0.3
        opt.setup_inexact_line_search(0,rho = rho, sigma = sigma,alpha_init=0.1)
        x = np.array([2])
        s = np.array([-1])
        alpha = opt.inexact_line_search(x,s)
        cond = wolfe_powell(f,gradf,x,s,alpha,rho,sigma)
        assert(cond == True)
        
    
    

unittest.main(argv=['first-arg-is-ignored'], exit=False)

x = np.array([2])
s = np.array([-1])
