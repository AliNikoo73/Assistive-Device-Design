"""
GaitSim Assist: A Python library for gait simulations and assistive device design.

This library provides tools for biomechanics researchers and wearable device engineers
to easily perform gait simulations, test various assistive device parameters, and
optimize cost function settings without deep diving into low-level OpenSim code.
"""

__version__ = "0.1.0"

from . import simulation
from . import cost_functions
from . import visualization
from . import analysis
from . import devices

# Convenience imports
from .simulation import GaitSimulator
from .devices import AssistiveDevice 