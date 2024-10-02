"""
Chebyquad Testproblem

Course material for the course FMNN25
Version for Python 3.4
Claus FÃ¼hrer (2016)

"""
#pip install scipy
import numpy as np
from  numpy import linspace, dot
import scipy.optimize as so
from numpy import array


def T(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the first kind
    x evaluation point (scalar)
    n degree
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x
    return 2. * x * T(x, n - 1) - T(x, n - 2)

def U(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the second kind
    x evaluation point (scalar)
    n degree
    Note d/dx T(x,n)= n*U(x,n-1)
    """
    if n == 0:
        return 1.0
    if n == 1:
        return 2. * x
    return 2. * x * U(x, n - 1) - U(x, n - 2)

def chebyquad_fcn(x):
    """
    Nonlinear function: R^n -> R^n
    """
    n = len(x)
    def exact_integral(n):
        """
        Generator object to compute the exact integral of
        the transformed Chebychev function T(2x-1,i), i=0...n
        """
        for i in range(n):
            if i % 2 == 0:
                yield -1./(i**2 - 1.)
            else:
                yield 0.

    exint = exact_integral(n)

    def approx_integral(i):
        """
        Approximates the integral by taking the mean value
        of n sample points
        """
        return sum(T(2. * xj - 1., i) for xj in x) / n
    return array([approx_integral(i) - e for i,e in enumerate(exint)])

def chebyquad(x):
    """
    norm(chebyquad_fcn)**2
    """
    chq = chebyquad_fcn(x)
    return dot(chq, chq)

def gradchebyquad(x):
    """
    Evaluation of the gradient function of chebyquad
    """
    chq = chebyquad_fcn(x)
    UM = 4. / len(x) * array([[(i+1) * U(2. * xj - 1., i)
                             for xj in x] for i in range(len(x) - 1)])
    return dot(chq[1:].reshape((1, -1)), UM).reshape((-1, ))

import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    try:
        from Project2.optimization import OptimizationProblem
        from Project2.optimizer import *
    except ModuleNotFoundError:
        from optimization import OptimizationProblem
        from optimizer import *
        
    def solve_and_summary(optimizer,x0,max_iter = 30):
        optimizer.solve(x,max_iter)
        optimizer.summary(sort_x = True)
        
        
    #n=4
    print("n=4")
    x=linspace(0,1,4)
    epsilon = 1e-8
    print("Using fmin_bfgs:")
    xmin4= so.fmin_bfgs(chebyquad,x,gradchebyquad)  # should converge after 18 iterations

    print("x=",xmin4)
    #chebyquad opt

    problem = OptimizationProblem(chebyquad, gradf = gradchebyquad)
    optimizer_GB4= GoodBroyden(problem,epsilon,line_search = "exact")
    optimizer_BB4= BadBroyden(problem,epsilon,line_search="exact")
    optimizer_SB4= SymmetricBroyden(problem,epsilon,line_search="exact")
    optimizer_DFP4= DFP(problem,epsilon,line_search="exact")
    optimizer_BFGS4= BFGS(problem,epsilon,line_search="exact")


    solve_and_summary(optimizer_GB4,x,30)
    solve_and_summary(optimizer_BB4,x,30)
    solve_and_summary(optimizer_SB4,x,30)
    solve_and_summary(optimizer_DFP4,x,30)
    solve_and_summary(optimizer_BFGS4,x,30)

    
    

  #n=8
    print("n=8")
    x=linspace(0,1,8)
    print("Using fmin_bfgs")
    xmin8= so.fmin_bfgs(chebyquad,x,gradchebyquad)  # should converge after 18 iterations
    print("x = ",np.sort(xmin8))
    #chebyquad opt
    epsilon = 1e-8
    problem = OptimizationProblem(chebyquad, gradf = gradchebyquad)
    optimizer_GB8= GoodBroyden(problem,epsilon,line_search="exact")
    optimizer_BB8= BadBroyden(problem,epsilon,line_search="exact")
    optimizer_SB8= SymmetricBroyden(problem,epsilon,line_search="exact")
    optimizer_DFP8= DFP(problem,epsilon,line_search="exact")
    optimizer_BFGS8= BFGS(problem,epsilon,line_search="exact")


    solve_and_summary(optimizer_GB8,x,30)
    solve_and_summary(optimizer_BB8,x,30)
    solve_and_summary(optimizer_SB8,x,30)
    solve_and_summary(optimizer_DFP8,x,30)
    solve_and_summary(optimizer_BFGS8,x,30)




    #n=11
    print("n=11")
    print("Using fmin_bfgs")
    x=linspace(0,1,11)
    xmin11= so.fmin_bfgs(chebyquad,x,gradchebyquad)  # should converge after 18 iterations

    print("x:",np.sort(xmin11))
    #chebyquad opt

    problem = OptimizationProblem(chebyquad, gradf = gradchebyquad)
    optimizer_GB11= GoodBroyden(problem,epsilon,line_search="exact")
    optimizer_BB11= BadBroyden(problem,epsilon,line_search="exact")
    optimizer_SB11= SymmetricBroyden(problem,epsilon,line_search="exact")
    optimizer_DFP11=DFP(problem,epsilon,line_search="exact")
    optimizer_BFGS11= BFGS(problem,epsilon,line_search="inexact",hessian_init="fd")
    optimizer_BFGS11.setup_inexact_line_search(0,rho = 0.01, sigma = 0.1)
    optimizer_newton = NewtonOptimizer(problem,line_search="exact")
    optimizer_newton.setup_inexact_line_search(0)
    


    solve_and_summary(optimizer_GB11,x,60)
    solve_and_summary(optimizer_BB11,x,60)
    solve_and_summary(optimizer_SB11,x,60)
    solve_and_summary(optimizer_DFP11,x,60)
    solve_and_summary(optimizer_BFGS11,x,60)
    solve_and_summary(optimizer_newton,x,60)


