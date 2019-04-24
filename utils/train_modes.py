"""
Module for defining training modes (e.g. training incrementally or accumulating the data of multiple megabatches)
"""
from enum import Enum


class TrainMode(Enum):
    """
    Enum for multiple training modes
    """
    INCREMENTAL = 0
    """
    Used when the data is presented in an strictly incremental way, that is, the megabatch 0 is presented, then
    megabatch 1, ..., and the data of a previously seen is not including for future training. This mode does not exclude
    the possibility of iterating multiple times over a single megabatch.
    """
    ACUMULATIVE = 1
    """
    Used when the data is presented in an acumulative way, that is, the megabatch 0 is presented, then megabatch 0 +
    megabatch 1, then megabatch 0 + megabatch 1 + megabatch 2, ...
    """
