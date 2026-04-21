class GaussianCondOTProbabilityPath:
    def __init__(self):
        pass

    def alpha(self, t):
        return t 
    
    def beta(self, t):
        return 1 - t
    
    def __call__(self, z, t, noise):
        return self.alpha(t) * z + self.beta(t) * noise