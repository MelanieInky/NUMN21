"""
Quadrature functions that should be useful.
"""
import numpy as np


def quadrature(f,x_list,w,c,t1,t2,gradf = None):
    """Generic formula to estimate the integral from t1 to t2 of f(x,t)

    Args:
        - f (_type_): The function f(x,t)
        - x_list: A list with elements being the values of x at the quadrature points
        - w (_type_): The associated weights
        - c (_type_): Vector used to offset the time values
        - t1 (_type_): Lower time bound of the integral
        - t2 (_type_): Upper time bound of the integral
        - gradf: Gradient of f w.r.t x, return the gradient w.r.t the x coordinate as well,
        if specified.
        
        
    The function is written in such a way to facilitate treating the values of x as unknown.
    
    Example w and c:
    - Midpoint: w = 1, c = 1/2
    - Trapezoidal w = [1/2,1/2], c = [0,1]
    - Simpson rule  w = [1/6, 4/6, 1/6], c = [0,1/2,1]
    """
    h = t2-t1
    w = np.squeeze(w) #making sure there are no empty dimensions
    c = np.squeeze(c)
    n_points = x_list.shape[0]
    if(len(w.shape) != 1 or len(c.shape) != 1):
        raise Exception("w and c should be one dimensional")
    if(w.shape[0] != n_points or c.shape[0] != n_points):
        raise Exception("w and c should have the same lengths equal to the one of x_list")
    quad = 0
    for i in range(len(c)):
        quad+= w[i] * f(x_list[i],t1 + c[i]*h) 
    quad *= h
    return quad




if __name__=='__main__':
    def f(x,t):
        return t
    x_list = np.zeros((2,3))
    w = [1/2,1/2]
    c = [0,1]
    test = quadrature(f,x_list,w,c,1,2)
    print(f'Testing trap rule: exp: 1.5, got = {test}') #Should be 1.5
    
    def g(x,t):
        return t**2

    #Simpson rule should integrate g exactly
    t1 = 5
    t2 = 9
    x_list = np.zeros((3,12))
    w = [1/6,4/6,1/6]
    c = [0,0.5,1]
    test = quadrature(g,x_list,w,c,t1,t2)
    print(f'Testing simpson rule: exp: {t2**3/3-t1**3/3}, got = {test}') #Should be 1.5
    
    
