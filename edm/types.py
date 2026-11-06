"""The eight MEDseq model types (Murphy et al., 2021, JRSS-A, Section 3.3).

First letter  : precision constrained (C) / unconstrained (U) across clusters.
Second letter : precision constrained (C) / unconstrained (U) across time points.
Trailing N    : an extra uniform noise component whose precision is fixed at 0.
"""
from enum import Enum

EDMType = Enum(
    "EDMType",
    [
        'CC',
        'UC',
        'CU',
        'UU',
        'CCN',
        'UCN',
        'CUN',
        'UUN',
    ]
)