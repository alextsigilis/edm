"""UUN: UU plus a uniform noise component (last index, λ = 0, f_0(s) = v^{-T}).

A precision for every (non-noise cluster, time point) pair. This is the type the paper
selects for the MVAD data (G = 11, GCSE5eq in a non-gated-noise gate). With G = 2 it is
identical to CUN.
"""
from .types import EDMType
from .uu import UUModel


class UUNModel(UUModel):
    edm_type = EDMType.UUN
    has_noise = True