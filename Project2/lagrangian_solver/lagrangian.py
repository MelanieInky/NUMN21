class LagrangianProblem:
    def __init__(self,K,V,grad_K,grad_V) -> None:
        """Class for hopefully solving physics problem with Hamilton principle
        
        The potential and kinetic energy functions  have the form f(x,t):R^n -> R. 
        The gradients have the form f'(x,t):R^n -> R^n
        """
        self.K = K #Kinetic energy function
        self.V = V #Potential energy function
        self.grad_K = grad_K #Gradient of the kinetic energy function
        self.grad_V = grad_V #Gradient of the potential energy function
        def L(x,t):
            return self.K(x,t) - self.V(x,t) #Lagrangian
        self.L = L
        def grad_L(x,t):
            return self.grad_K(x,t) - self.grad_V(x,t)
        self.grad_L = grad_L
        pass
    
    def S(self,X_list,T_list): #Action
        #IDEA, use a quadrature formula
        pass
        
    
    