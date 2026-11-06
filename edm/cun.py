"""CUN: CU plus a uniform noise component (last index, λ = 0, f_0(s) = v^{-T}).

One precision per time point, shared by the G - 1 non-noise clusters.
"""
from .cu import CUModel
from .types import EDMType


class CUNModel(CUModel):
    edm_type = EDMType.CUN
    has_noise = True