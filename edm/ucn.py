"""UCN: UC plus a uniform noise component (last index, λ = 0, f_0(s) = v^{-T}).

One precision per non-noise cluster. With G = 2 (a single non-noise component) this is
identical to CCN.
"""
from .types import EDMType
from .uc import UCModel


class UCNModel(UCModel):
    edm_type = EDMType.UCN
    has_noise = True