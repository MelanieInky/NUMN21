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
  #n=8
    x=linspace(0,1,8)
    xmin8= so.fmin_bfgs(chebyquad,x,gradchebyquad)  # should converge after 18 iterations
    print("n=8")
    print("Using fmin_bfgs:",np.sort(xmin8))
    #chebyquad opt
    epsilon = 1e-2
    problem = OptimizationProblem(chebyquad, gradf = gradchebyquad)
    optimizer_GB8= GoodBroyden(problem,epsilon,line_search="exact")
    optimizer_BB8= BadBroyden(problem,epsilon,line_search="exact")
    optimizer_SB8= SymmetricBroyden(problem,epsilon,line_search="exact")
    optimizer_DFP8= DFP(problem,epsilon,line_search="exact")
    optimizer_BFGS8= BFGS(problem,epsilon,line_search="exact")


    optimizer_GB8.solve(x,30)
    optimizer_BB8.solve(x,30)
    optimizer_SB8.solve(x,30)
    optimizer_DFP8.solve(x,30)
    optimizer_BFGS8.solve(x,30)




    print(f'optimizer_GB8.xhist: {np.sort(optimizer_GB8.xhist[-1])}')
    print(f'optimizer_BB8.xhist: {np.sort(optimizer_BB8.xhist[-1])}')
    print(f'optimizer_SB8.xhist: {np.sort(optimizer_SB8.xhist[-1])}')
    print(f'optimizer_DFP8.xhist: {np.sort(optimizer_DFP8.xhist[-1])}')
    print(f'optimizer_BFGS8.xhist: {np.sort(optimizer_BFGS8.xhist[-1])}')

  #n=4
    x=linspace(0,1,4)
    xmin4= so.fmin_bfgs(chebyquad,x,gradchebyquad)  # should converge after 18 iterations
    print("n=4")
    print("Using fmin_bfgs:",xmin4)
    #chebyquad opt

    problem = OptimizationProblem(chebyquad, gradf = gradchebyquad)
    optimizer_GB4= GoodBroyden(problem,epsilon,line_search = "exact")
    optimizer_BB4= BadBroyden(problem,epsilon,line_search="exact")
    optimizer_SB4= SymmetricBroyden(problem,epsilon,line_search="exact")
    optimizer_DFP4= DFP(problem,epsilon,line_search="exact")
    optimizer_BFGS4= BFGS(problem,epsilon,line_search="exact")


    optimizer_GB4.solve(x,30)
    optimizer_BB4.solve(x,30)
    optimizer_SB4.solve(x,30)
    optimizer_DFP4.solve(x,30)
    optimizer_BFGS4.solve(x,30)




    print(f'optimizer_GB4.xhist: {np.sort(optimizer_GB4.xhist[-1])}')
    print(f'optimizer_BB4.xhist: {np.sort(optimizer_BB4.xhist[-1])}')
    print(f'optimizer_SB4.xhist: {np.sort(optimizer_SB4.xhist[-1])}')
    print(f'optimizer_DFP4.xhist: {np.sort(optimizer_DFP4.xhist[-1])}')
    print(f'optimizer_BFGS4.xhist: {np.sort(optimizer_BFGS4.xhist[-1])}')
      #n=11
    x=linspace(0,1,11)
    xmin11= so.fmin_bfgs(chebyquad,x,gradchebyquad)  # should converge after 18 iterations
    print("n=11")
    print("Using fmin_bfgs:",np.sort(xmin11))
    #chebyquad opt

    problem = OptimizationProblem(chebyquad, gradf = gradchebyquad)
    optimizer_GB11= GoodBroyden(problem,epsilon,line_search="exact")
    optimizer_BB11= BadBroyden(problem,epsilon,line_search="exact")
    optimizer_SB11= SymmetricBroyden(problem,epsilon,line_search="exact")
    optimizer_DFP11=DFP(problem,epsilon,line_search="exact")
    optimizer_BFGS11= BFGS(problem,epsilon,line_search="exact")


    optimizer_GB11.solve(x,30)
    optimizer_BB11.solve(x,30)
    optimizer_SB11.solve(x,30)
    optimizer_DFP11.solve(x,30)
    optimizer_BFGS11.solve(x,30)




    print(f'optimizer_GB11.xhist: {np.sort(optimizer_GB11.xhist[-1])}')
    print(f'optimizer_BB11.xhist: {np.sort(optimizer_BB11.xhist[-1])}')
    print(f'optimizer_SB11.xhist: {np.sort(optimizer_SB11.xhist[-1])}')
    print(f'optimizer_DFP11.xhist: {np.sort(optimizer_DFP11.xhist[-1])}')
    print(f'optimizer_BFGS11.xhist: {np.sort(optimizer_BFGS11.xhist[-1])}')
