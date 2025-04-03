"""
Input Scaling Utilities for Kolmogorov-Arnold Networks

This module provides scaling utilities for preprocessing input data before feeding it
into Kolmogorov-Arnold Networks. Properly scaled inputs can significantly improve
training stability and model performance.
"""

import torch
import torch.nn as nn
import numpy as np


class BaseScaler(nn.Module):
    """
    Abstract base class for all scalers used with Kolmogorov-Arnold Networks.
    
    Scalers transform input features to a standardized range to improve
    training stability and convergence. All concrete scaler implementations
    should inherit from this class.
    """
    def __init__(self):
        super(BaseScaler, self).__init__()
        
    def fit(self, x):
        """
        Compute the scaling parameters from the input data.
        
        Args:
            x (torch.Tensor or numpy.ndarray): Input data to compute scaling parameters from.
            
        Returns:
            self: The fitted scaler object.
        """
        raise NotImplementedError("Subclasses must implement fit method")
        
    def forward(self, x):
        """
        Apply the scaling transformation to the input data.
        
        Args:
            x (torch.Tensor): Input tensor to be scaled.
            
        Returns:
            torch.Tensor: Scaled input tensor.
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def inverse_transform(self, x):
        """
        Apply the inverse scaling transformation to recover original scale.
        
        Args:
            x (torch.Tensor): Scaled tensor to be transformed back.
            
        Returns:
            torch.Tensor: Tensor in the original scale.
        """
        raise NotImplementedError("Subclasses must implement inverse_transform method")


class MinMaxScaler(BaseScaler):
    """
    Scales features to a specified range, typically [0, 1] or [-1, 1].
    
    MinMaxScaler transforms features by scaling each feature to a given range.
    This is done by subtracting the minimum value of the feature and then dividing
    by the range (max - min).
    
    Attributes:
        feature_range (tuple): Min and max values of the target scaling range.
        min_vals (torch.Tensor): Minimum values of each feature in the original data.
        max_vals (torch.Tensor): Maximum values of each feature in the original data.
        scale (torch.Tensor): Scaling factor for each feature (max - min).
    """
    def __init__(self, feature_range=(0, 1)):
        """
        Initialize a MinMaxScaler.
        
        Args:
            feature_range (tuple, optional): The desired range of the transformed data. 
                                            Defaults to (0, 1).
        """
        super(MinMaxScaler, self).__init__()
        self.feature_range = feature_range
        self.register_buffer('min_vals', None)
        self.register_buffer('max_vals', None)
        self.register_buffer('scale', None)
        
    def fit(self, x):
        """
        Compute the minimum and maximum values for scaling.
        
        Args:
            x (torch.Tensor or numpy.ndarray): Data to compute scaling parameters from.
            
        Returns:
            self: The fitted scaler object.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            
        min_vals = torch.min(x, dim=0)[0]
        max_vals = torch.max(x, dim=0)[0]
        
        # Handle case where min and max are identical
        scale = max_vals - min_vals
        scale[scale == 0] = 1.0
        
        self.register_buffer('min_vals', min_vals)
        self.register_buffer('max_vals', max_vals)
        self.register_buffer('scale', scale)
        
        return self
        
    def forward(self, x):
        """
        Scale features to the specified range.
        
        Args:
            x (torch.Tensor): Input tensor to scale.
            
        Returns:
            torch.Tensor: Scaled input tensor.
            
        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if self.min_vals is None or self.max_vals is None:
            raise RuntimeError("MinMaxScaler has not been fitted. Call fit() first.")
            
        x_std = (x - self.min_vals) / self.scale
        x_scaled = x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        return x_scaled
    
    def inverse_transform(self, x):
        """
        Inverse transform to recover original scale.
        
        Args:
            x (torch.Tensor): Scaled tensor to transform back.
            
        Returns:
            torch.Tensor: Tensor in the original scale.
            
        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if self.min_vals is None or self.max_vals is None:
            raise RuntimeError("MinMaxScaler has not been fitted. Call fit() first.")
            
        x_std = (x - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        x_original = x_std * self.scale + self.min_vals
        
        return x_original


class StandardScaler(BaseScaler):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    
    StandardScaler transforms features to have zero mean and unit variance,
    which is a common requirement for many machine learning algorithms.
    
    Attributes:
        mean (torch.Tensor): Mean values of each feature in the original data.
        std (torch.Tensor): Standard deviation values of each feature in the original data.
        epsilon (float): Small constant to prevent division by zero.
    """
    def __init__(self, epsilon=1e-10):
        """
        Initialize a StandardScaler.
        
        Args:
            epsilon (float, optional): Small constant to prevent division by zero. 
                                      Defaults to 1e-10.
        """
        super(StandardScaler, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
        
    def fit(self, x):
        """
        Compute the mean and standard deviation for scaling.
        
        Args:
            x (torch.Tensor or numpy.ndarray): Data to compute scaling parameters from.
            
        Returns:
            self: The fitted scaler object.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        
        # Handle features with zero standard deviation
        std[std < self.epsilon] = 1.0
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        return self
        
    def forward(self, x):
        """
        Standardize features by removing the mean and scaling to unit variance.
        
        Args:
            x (torch.Tensor): Input tensor to standardize.
            
        Returns:
            torch.Tensor: Standardized input tensor.
            
        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted. Call fit() first.")
            
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x):
        """
        Inverse transform to recover original scale.
        
        Args:
            x (torch.Tensor): Standardized tensor to transform back.
            
        Returns:
            torch.Tensor: Tensor in the original scale.
            
        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted. Call fit() first.")
            
        return x * self.std + self.mean


class IdentityScaler(BaseScaler):
    """
    A pass-through scaler that does not modify the input data.
    
    This scaler is useful when you need to maintain the interface consistency
    but don't want to apply any scaling to the data.
    """
    def __init__(self):
        """Initialize an IdentityScaler."""
        super(IdentityScaler, self).__init__()
        
    def fit(self, x):
        """
        No-op method to maintain the interface consistency.
        
        Args:
            x (torch.Tensor or numpy.ndarray): Input data (not used).
            
        Returns:
            self: The scaler object (unchanged).
        """
        return self
        
    def forward(self, x):
        """
        Pass the input through unchanged.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: The same input tensor, unchanged.
        """
        return x
    
    def inverse_transform(self, x):
        """
        Pass the input through unchanged.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: The same input tensor, unchanged.
        """
        return x