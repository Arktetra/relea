from abc import ABC

class ProbabilityPath(ABC):
    def __init__(self):
        pass

    def alpha(self, t):
        raise NotImplementedError("implement me!")
    
    def beta(self, t):
        raise NotImplementedError("implement me!")
    