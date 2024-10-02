#from Project2 import finite_diff
from finite_diff import finite_difference
from collections.abc import Callable
import numpy.typing as npt
from typing import Concatenate

class OptimizationProblem:
    """Class that defines an optimization problem
    """
    def __init__(self,f: Callable[Concatenate[npt.ArrayLike,...],npt.ArrayLike],gradf=None, **kwargs):
        """Defines an optimization problem to find the minimum of the function f
        
        Args:
            f (Callable[Concatenate[npt.ArrayLike,...],float]): The function f(x,**kwargs) to minimize, from R^n -> R
            gradf (Callable, optional): Optionally defines the Jacobian of the function. If not given, computes the Jacobian via finite differences. Defaults to None.
            **kwargs: Additional arguments to be passed to the function.
        Returns:
            An OptimizationProblem object to be passed to the Optimizer class
        """
        self.f =f
        self.kwargs = kwargs
        self.epsilon = 1e-6
        if gradf is None:
            def gradf(x,**kwargs):
                return finite_difference(f,x,self.epsilon,**kwargs).flatten()
            self.gradf = gradf
        else:
            self.gradf = gradf
        pass
    
    def optimize(self,optimizer,x0,max_iter):
        """Optimize the function using the optimizer given in argument

        Args:
            optimizer (_type_): An Optimizer object
            x0 (_type_): Initial guess
            max_iter (_type_): Maximum number of iteration before giving up
        """
        pass
    
    

