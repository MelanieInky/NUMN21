""" Tests for the different line searches"""
import unittest

from Project2.finite_diff import finite_difference
from Project2.optimization import OptimizationProblem
from Project2.optimizer import NewtonOptimizer


def f(x):
    return x**2

def gradf(x):
    return 2*x


class TestANN(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.problem = OptimizationProblem(f,gradf)
        self.opt = NewtonOptimizer(self.problem)

    def test_line_search(self):
        x = 2
        s = -1
        alpha = self.opt.exact_line_search(x,s)
        print(alpha)
        self.assertAlmostEqual(alpha,2)
        
        

unittest.main(argv=['first-arg-is-ignored'], exit=False)