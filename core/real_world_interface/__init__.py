"""
Real-World Interface Layer
Bridges simulations with external real-world data and computational tools
"""

from .data_connectors import RealWorldDataConnector
from .computational_executors import ComputationalExecutor
from .safety_validators import RealWorldSafetyValidator

__all__ = [
    'RealWorldDataConnector',
    'ComputationalExecutor', 
    'RealWorldSafetyValidator'
]