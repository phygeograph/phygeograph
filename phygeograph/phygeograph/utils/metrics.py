import numpy as np

def rmse(predictions, targets):
    """ RMSE function  
       Args:
           predictions (1-d array of float): Predicted values.
           targets (1-d array of float): Observed values.
    """
    return np.sqrt(((predictions - targets) ** 2).mean())
