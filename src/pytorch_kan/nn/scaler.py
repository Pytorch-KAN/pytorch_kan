"""Data scaling utilities for Kolmogorov-Arnold Networks.

This module provides scalers for normalizing input data before feeding it to KANs.
Proper scaling is crucial for the performance of KANs, as it ensures inputs are
within a suitable range for the basis functions.
"""

import torch
import torch.nn as nn


class MinMaxScaler(nn.Module):
    """
    Min-Max scaler that normalizes data to the [0, 1] range.
    
    This scaler transforms each feature by scaling it to a given range, 
    typically [0, 1], based on the minimum and maximum values of the feature.
    
    Attributes:
        min (torch.Tensor): Minimum values for each feature.
        max (torch.Tensor): Maximum values for each feature.
        feature_range (tuple): The output range for scaled data, default (0, 1).
    """
    
    def __init__(self, feature_range=(0, 1)):
        """
        Initialize a MinMaxScaler.
        
        Args:
            feature_range (tuple, optional): The output range for scaled data. Defaults to (0, 1).
        """
        super(MinMaxScaler, self).__init__()
        self.feature_range = feature_range
        self.min = None
        self.max = None
        self.register_buffer('min_val', None)
        self.register_buffer('max_val', None)
    
    def fit(self, x):
        """
        Compute the minimum and maximum values for each feature.
        
        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, n_features].
            
        Returns:
            MinMaxScaler: The fitted scaler.
        """
        self.min_val = torch.min(x, dim=0)[0]
        self.max_val = torch.max(x, dim=0)[0]
        self.min = self.min_val  # For API compatibility with test
        self.max = self.max_val  # For API compatibility with test
        return self
    
    def transform(self, x):
        """
        Scale the data to the [0, 1] range.
        
        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, n_features].
            
        Returns:
            torch.Tensor: Scaled data tensor of the same shape.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted. Call fit() before transform().")
        
        # Prevent division by zero
        denom = self.max_val - self.min_val
        denom[denom == 0] = 1.0
        
        # Scale to [0, 1]
        scaled = (x - self.min_val) / denom
        
        # Scale to feature_range if different from [0, 1]
        if self.feature_range != (0, 1):
            scaled = scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        return scaled
    
    def inverse_transform(self, x):
        """
        Reverse the scaling transformation.
        
        Args:
            x (torch.Tensor): Scaled data tensor of shape [batch_size, n_features].
            
        Returns:
            torch.Tensor: Original-scale data tensor of the same shape.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted. Call fit() before inverse_transform().")
        
        # Scale back from feature_range if different from [0, 1]
        if self.feature_range != (0, 1):
            x = (x - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        
        # Scale back to original range
        return x * (self.max_val - self.min_val) + self.min_val
    
    def fit_transform(self, x):
        """
        Fit the scaler and transform the data in one step.
        
        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, n_features].
            
        Returns:
            torch.Tensor: Scaled data tensor of the same shape.
        """
        return self.fit(x).transform(x)


class StandardScaler(nn.Module):
    """
    Standard scaler that normalizes data to have zero mean and unit variance.
    
    This scaler transforms each feature by subtracting the mean and dividing by
    the standard deviation, resulting in a distribution with mean 0 and standard
    deviation 1.
    
    Attributes:
        mean (torch.Tensor): Mean values for each feature.
        std (torch.Tensor): Standard deviation values for each feature.
    """
    
    def __init__(self):
        """Initialize a StandardScaler."""
        super(StandardScaler, self).__init__()
        self.mean = None
        self.std = None
        self.register_buffer('mean_val', None)
        self.register_buffer('std_val', None)
    
    def fit(self, x):
        """
        Compute the mean and standard deviation for each feature.
        
        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, n_features].
            
        Returns:
            StandardScaler: The fitted scaler.
        """
        self.mean_val = torch.mean(x, dim=0)
        self.std_val = torch.std(x, dim=0, unbiased=False)
        self.mean = self.mean_val  # For API compatibility with test
        self.std = self.std_val  # For API compatibility with test
        return self
    
    def transform(self, x):
        """
        Scale the data to have zero mean and unit variance.
        
        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, n_features].
            
        Returns:
            torch.Tensor: Scaled data tensor of the same shape.
        """
        if self.mean_val is None or self.std_val is None:
            raise ValueError("Scaler has not been fitted. Call fit() before transform().")
        
        # Prevent division by zero
        std = self.std_val.clone()
        std[std == 0] = 1.0
        
        return (x - self.mean_val) / std
    
    def inverse_transform(self, x):
        """
        Reverse the scaling transformation.
        
        Args:
            x (torch.Tensor): Scaled data tensor of shape [batch_size, n_features].
            
        Returns:
            torch.Tensor: Original-scale data tensor of the same shape.
        """
        if self.mean_val is None or self.std_val is None:
            raise ValueError("Scaler has not been fitted. Call fit() before inverse_transform().")
        
        # Prevent division by zero
        std = self.std_val.clone()
        std[std == 0] = 1.0
        
        return x * std + self.mean_val
    
    def fit_transform(self, x):
        """
        Fit the scaler and transform the data in one step.
        
        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, n_features].
            
        Returns:
            torch.Tensor: Scaled data tensor of the same shape.
        """
        return self.fit(x).transform(x)


class IdentityScaler(nn.Module):
    """
    Identity scaler that does not modify the data.
    
    This scaler is useful as a placeholder when no scaling is desired, while
    maintaining a consistent API with other scalers.
    """
    
    def __init__(self):
        """Initialize an IdentityScaler."""
        super(IdentityScaler, self).__init__()
    
    def fit(self, x):
        """
        No-op function to maintain API compatibility.
        
        Args:
            x (torch.Tensor): Input data tensor.
            
        Returns:
            IdentityScaler: The scaler itself.
        """
        return self
    
    def transform(self, x):
        """
        Return the data unchanged.
        
        Args:
            x (torch.Tensor): Input data tensor.
            
        Returns:
            torch.Tensor: The same input tensor.
        """
        return x
    
    def inverse_transform(self, x):
        """
        Return the data unchanged.
        
        Args:
            x (torch.Tensor): Input data tensor.
            
        Returns:
            torch.Tensor: The same input tensor.
        """
        return x
    
    def fit_transform(self, x):
        """
        Return the data unchanged.
        
        Args:
            x (torch.Tensor): Input data tensor.
            
        Returns:
            torch.Tensor: The same input tensor.
        """
        return x