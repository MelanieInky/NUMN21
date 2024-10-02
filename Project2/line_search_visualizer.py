"""
Visualization of the phi function for the line search
"""


import numpy as np
from collections.abc import Callable
import numpy.typing as npt
from typing import Concatenate
import matplotlib.pyplot as plt


def plot_phi(f: Callable[Concatenate[npt.ArrayLike,...],float],
             x: npt.ArrayLike,s: npt.ArrayLike,
             alpha_max = 4.,
             rho = 0.5,
             sigma = 0.9,
             **kwargs):
    """Plots phi(alpha) = f(x + alpha*s) to visualize the line search.

    Args:
        f (Callable[Concatenate[npt.ArrayLike,...],float]): The function f of the form f(x,**kwargs)
        x (np.ndarray): The value of x
        s (np.ndarray): The search direction
        rho (float,optional): rho parameter between 0 and 1/2. Defaults to 1/2
        sigma (float,optional): sigma parameter between rho and 1. Defaults to 0.9
        alpha_max (float, optional): Maximum value of alpha to plot for. Defaults to 4.
        **kwargs: Keywords arguments to be passed to the f.
    """
    def phi(alpha):
        return f(x + alpha*s,**kwargs)
    
    eps = 1e-7
    phi0 = phi(0)
    dphi0 = (phi(eps) - phi(0)) / eps #Finite difference
    
    alphas = np.linspace(0,alpha_max,400)
    rho_line = phi0 + alphas * rho * dphi0
    phi_val = np.zeros_like(alphas) #phi(alpha)
    dphi_val = np.zeros_like(alphas) #phi'(alpha)
    wp2_i = None #Index of the second wolfe powell condition
    gs1_i = None #Index of the Goldstein condition
    for i , alpha in enumerate(alphas):
        phi_val[i] = phi(alpha)
        dphi_val[i] = (phi(alpha + eps) - phi_val[i]) / eps
        #Check if wolfe powell second condition is broken
        if(wp2_i == None):
            wp2_cond = dphi_val[i] > sigma * dphi0
            if(wp2_cond): #Save the first time it gets broken
                wp2_i = i
                print(wp2_i)
        if(gs1_i == None):
            gs1_cond = phi_val[i] > phi0 + alpha * rho * dphi0
            if(gs1_cond):
                gs1_i = i
    alpha_neighb = np.linspace(alphas[wp2_i] - alpha_max/10,alphas[wp2_i] + alpha_max/10)
    wp2_line = phi_val[wp2_i] + dphi_val[wp2_i] * (alpha_neighb - alphas[wp2_i])
    
    fig, ax = plt.subplots()
    ax.plot(alphas,phi_val)
    ax.plot(alphas,rho_line,label = 'rho line',linestyle = '--', c = 'orange')
    #ax.plot(alpha_neighb,wp2_line,label = 'Wolfe-Powell condition 2',linestyle = '--',c = 'green')
    ax.plot(alpha_neighb,wp2_line,label = 'Condition',linestyle = '--',c = 'green')
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$\\phi(\\alpha)$")
    ax.legend()
    ax.grid()
    if(gs1_i is None or wp2_i is None):
        raise Exception("Could not find the acceptable bounds with the provided values of alpha, try increasing alpha_max")
    b = alphas[gs1_i]
    a = alphas[wp2_i]
    phi_val_acc= phi_val[wp2_i:gs1_i]
    i_best = np.argmin(phi_val_acc)
    print(alphas[i_best + wp2_i])
    ax.set_xticks([alphas[gs1_i],alphas[wp2_i],alphas[i_best+wp2_i]],
                  [f"b = {b:.2f}",f"a = {a:.2f}",f"{alphas[i_best+wp2_i]:.2f}"],
                  rotation = 90)
    fig.tight_layout()
    return fig, ax
    
if __name__ == '__main__':
    from Rosenbrock import Rosenbrock, gradRosenbrock
    from optimization import OptimizationProblem
    from optimizer import NewtonOptimizer
    
    def f(x,r):
        return x*(x + r)
    
    #Minimum at -1, direction is -1 so best alpha should be 2.
    #fig, ax =plot_phi(f,1,-1,sigma = 0.55,rho = 0.3,r = 2) 
    #plt.show()

    problem = OptimizationProblem(Rosenbrock, gradf = gradRosenbrock)
    optimizer = NewtonOptimizer(problem,1e-9,"none")
    x = np.array([2,3])
    s = np.array([-1,2])
    rho = 0.1
    sigma = 0.5
    optimizer.setup_inexact_line_search(0,rho = rho, sigma = sigma)
    alpha = optimizer.inexact_line_search(x,s)
    fig2, ax2 = plot_phi(Rosenbrock,x,s,alpha_max=0.5,rho = rho,sigma = sigma)
    ax2.axvline(x = alpha,linestyle = "--",c = 'black',label = "Alpha (inexact line search)")
    fig2.legend()
    plt.show()
    