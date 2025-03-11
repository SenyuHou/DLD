import torch.nn as nn

class EMA(object):
    """
    Exponential Moving Average (EMA) class to maintain a smoothed version of model parameters.

    Attributes:
    - mu: The decay rate for the moving average.
    - shadow: A dictionary to store the EMA of model parameters.
    """
    def __init__(self, mu=0.999):
        """
        Initialize the EMA class with the given decay rate.

        Parameters:
        - mu: The decay rate for the moving average (default is 0.999).
        """
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        """
        Register the parameters of a module for EMA.

        Parameters:
        - module: The module whose parameters are to be registered.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        """
        Update the EMA of the registered parameters.

        Parameters:
        - module: The module whose parameters are to be updated.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        """
        Apply the EMA to the module's parameters.

        Parameters:
        - module: The module whose parameters are to be updated with EMA values.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        """
        Create a copy of the module with EMA parameters.

        Parameters:
        - module: The module to be copied.

        Returns:
        - module_copy: A copy of the module with EMA parameters.
        """
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        """
        Get the state dictionary of the EMA parameters.

        Returns:
        - self.shadow: The state dictionary of the EMA parameters.
        """
        return self.shadow

    def load_state_dict(self, state_dict):
        """
        Load the EMA parameters from a state dictionary.

        Parameters:
        - state_dict: The state dictionary containing the EMA parameters.
        """
        self.shadow = state_dict
